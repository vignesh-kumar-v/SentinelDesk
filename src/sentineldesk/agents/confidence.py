"""Scoring how much to trust a draft resolution, and deciding whether to escalate.

Rule-based on purpose. The blueprint's own call, and the right one: a learned
confidence head would need its own labelled data and its own evaluation, and the
depth in this project belongs in the DPO work, not here. Rule-based also means the
escalation decision is auditable, which is what a support org actually needs -
"escalated because the draft contradicted ACC-02" beats "escalated because the head
emitted 0.41".

The signals are combined as penalties against a starting confidence of 1.0 rather
than as a learned weighting, so every deduction is traceable to a named cause.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..data.kb import get_policy

_WORD = re.compile(r"[a-z0-9']+")

HEDGES = (
    "i'm not sure", "i am not sure", "not certain", "i think", "maybe", "possibly",
    "i believe", "might be", "unclear", "as an ai", "i cannot help", "i don't know",
    "i do not know", "please contact support", "contact our support team",
)

# Requests where the policy itself forbids an autonomous answer. These escalate
# regardless of how confident the draft sounds - a fluent, confident reply is
# exactly the dangerous failure here, not a reassuring one.
MANDATORY_ESCALATION = {
    "ACC-02": ("disable 2fa", "disable two-factor", "turn off 2fa", "remove 2fa",
               "lost my recovery", "lost recovery codes"),
    "ACC-04": ("gdpr", "right to erasure", "erase all my data", "delete permanently"),
    "TEC-05": ("service credit", "sla credit", "compensate", "compensation for downtime"),
    "BIL-05": ("change the company name", "change billed entity", "reissue the invoice"),
}


@dataclass
class ConfidenceReport:
    score: float
    signals: dict[str, float | bool | int] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def repetition_ratio(text: str, n: int = 6) -> float:
    """Fraction of n-grams that are repeats.

    A small model that loses the plot mid-response loops on a phrase. That failure
    reads as fluent and confident to every other signal here, so it needs its own.
    """
    toks = _tokens(text)
    if len(toks) < n * 2:
        return 0.0
    grams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def policy_overlap(text: str, policy_ids: list[str]) -> float:
    """Share of the draft's content words that appear in the retrieved policies.

    A proxy for groundedness, not a measure of correctness: a response can overlap
    heavily and still be wrong. It catches the opposite failure - a draft that
    ignores the retrieved policy entirely and answers from the model's priors.
    """
    draft = set(_tokens(text))
    if not draft:
        return 0.0
    stop = {
        "the", "a", "an", "and", "or", "to", "of", "in", "for", "is", "are", "you",
        "your", "we", "our", "it", "this", "that", "on", "with", "will", "can", "be",
        "have", "has", "if", "at", "as", "not", "but", "from", "by", "i", "me", "my",
    }
    content = draft - stop
    if not content:
        return 0.0
    policy_words: set[str] = set()
    for pid in policy_ids:
        try:
            policy_words |= set(_tokens(get_policy(pid).body))
        except KeyError:
            continue
    return len(content & policy_words) / len(content)


def mandatory_escalation_hits(subject: str, body: str, policy_ids: list[str]) -> list[str]:
    text = f"{subject} {body}".lower()
    hits = []
    for pid, phrases in MANDATORY_ESCALATION.items():
        if pid in policy_ids and any(p in text for p in phrases):
            hits.append(pid)
    return hits


def score_draft(
    draft: str,
    *,
    policy_ids: list[str],
    triage_confidence: float,
    mean_logprob: float | None = None,
    min_words: int = 15,
    max_words: int = 220,
) -> ConfidenceReport:
    words = _tokens(draft)
    signals: dict[str, float | bool | int] = {}
    reasons: list[str] = []
    score = 1.0

    signals["n_words"] = len(words)
    if len(words) < min_words:
        score -= 0.5
        reasons.append(f"draft is only {len(words)} words")
    elif len(words) > max_words:
        score -= 0.15
        reasons.append(f"draft ran to {len(words)} words")

    rep = repetition_ratio(draft)
    signals["repetition_ratio"] = round(rep, 3)
    if rep > 0.25:
        score -= 0.4
        reasons.append(f"repetitive output ({rep:.0%} repeated n-grams)")

    hedges = [h for h in HEDGES if h in draft.lower()]
    signals["hedges"] = len(hedges)
    if hedges:
        score -= min(0.15 * len(hedges), 0.45)
        reasons.append(f"hedged: {hedges[0]!r}")

    overlap = policy_overlap(draft, policy_ids)
    signals["policy_overlap"] = round(overlap, 3)
    if overlap < 0.12:
        score -= 0.35
        reasons.append(f"draft barely overlaps the retrieved policy ({overlap:.0%})")

    signals["triage_confidence"] = round(triage_confidence, 3)
    if triage_confidence < 0.5:
        # Low triage confidence means the retrieved policies may be the wrong ones,
        # so the draft's groundedness signal above is measuring the wrong target.
        score -= 0.2
        reasons.append(f"triage was unsure ({triage_confidence:.2f})")

    if mean_logprob is not None:
        signals["mean_logprob"] = round(mean_logprob, 3)
        if mean_logprob < -1.5:
            score -= 0.2
            reasons.append(f"low mean token logprob ({mean_logprob:.2f})")

    return ConfidenceReport(score=max(0.0, min(score, 1.0)), signals=signals, reasons=reasons)
