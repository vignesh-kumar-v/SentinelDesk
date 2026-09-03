"""Phase 1 verification: is the judge's labelling trustworthy enough to train on?

Two independent checks, because they answer different questions and neither alone
is enough:

1. Second-judge agreement. A *different* frontier model re-runs the same rubric on
   the same comparisons, in both orders. Cheap, unbiased with respect to the person
   who wrote the rubric, and it separates "the rubric is ambiguous" from "this
   particular model is quirky". Reported as raw agreement plus Cohen's kappa.

2. Human spot-check. The blueprint's own gate: a person labels 20 comparisons blind
   and we measure how often they disagree with the judge. This is the check that
   catches a rubric that is internally consistent but wrong, which no amount of
   model-vs-model agreement can detect. `sentineldesk spotcheck-human` runs it.

Kappa rather than raw agreement because these labels are heavily skewed - if the
judge picks the grounded strategy 85% of the time, a second labeller who always
picked grounded would score 85% raw agreement while carrying no information.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ..data.schema import Candidate, Ticket, dump_json
from ..llm import ChatClient
from ..logging_utils import get_logger
from .judge import Judge, PairJudgement

log = get_logger(__name__)


@dataclass
class AgreementResult:
    n: int
    agree: int
    raw_agreement: float
    cohens_kappa: float
    confusion: dict[str, int]
    disagreements: list[dict]

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "agree": self.agree,
            "raw_agreement": round(self.raw_agreement, 4),
            "cohens_kappa": round(self.cohens_kappa, 4),
            "confusion": self.confusion,
            "disagreements": self.disagreements,
        }


def cohens_kappa(a: list[str], b: list[str]) -> float:
    """Unweighted Cohen's kappa for two label sequences over the same items."""
    if not a:
        return 0.0
    n = len(a)
    labels = sorted(set(a) | set(b))
    po = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[k] / n) * (cb[k] / n) for k in labels)
    if pe >= 1.0:
        # Both labellers were constant and identical: agreement is total but the
        # statistic is undefined. Reporting 1.0 here would overstate it.
        return float("nan")
    return (po - pe) / (1 - pe)


def _compare(primary: list[str], other: list[str], meta: list[dict]) -> AgreementResult:
    agree = sum(1 for x, y in zip(primary, other, strict=True) if x == y)
    confusion = Counter(f"{x}|{y}" for x, y in zip(primary, other, strict=True))
    disagreements = [
        m | {"primary": x, "other": y}
        for x, y, m in zip(primary, other, meta, strict=True)
        if x != y
    ]
    return AgreementResult(
        n=len(primary),
        agree=agree,
        raw_agreement=agree / len(primary) if primary else 0.0,
        cohens_kappa=cohens_kappa(primary, other),
        confusion=dict(confusion),
        disagreements=disagreements,
    )


def sample_for_check(
    judgements: list[PairJudgement], n: int, seed: int
) -> list[PairJudgement]:
    """Sample comparisons to re-check.

    Stratified by outcome so the sample is not 20 easy wins for the same strategy;
    the disagreements worth finding cluster in the ties and the narrow margins.
    """
    rng = random.Random(seed)
    buckets: dict[str, list[PairJudgement]] = {}
    for j in judgements:
        key = j.winner if j.consistent else "tie"
        buckets.setdefault(key, []).append(j)
    for group in buckets.values():
        rng.shuffle(group)

    picked: list[PairJudgement] = []
    keys = sorted(buckets)
    i = 0
    while len(picked) < min(n, len(judgements)):
        group = buckets[keys[i % len(keys)]]
        if group:
            picked.append(group.pop())
        i += 1
        if all(not buckets[k] for k in keys):
            break
    return picked


def second_judge_agreement(
    judgements: list[PairJudgement],
    tickets: dict[str, Ticket],
    candidates: dict[tuple[str, str], Candidate],
    second: Judge,
    *,
    n: int = 40,
    seed: int = 1337,
    concurrency: int = 4,
) -> AgreementResult:
    sample = sample_for_check(judgements, n, seed)
    items = []
    for j in sample:
        strategies = sorted(j.scores)
        items.append(
            (tickets[j.ticket_id], candidates[(strategies[0], j.ticket_id)], candidates[(strategies[1], j.ticket_id)])
        )
    log.info("second-judge cross-check: %d comparisons via %s", len(items), second.model)
    others = second.judge_many(items, concurrency=concurrency)

    primary_labels, other_labels, meta = [], [], []
    for j, o in zip(sample, others, strict=True):
        if o is None:
            continue
        primary_labels.append(j.winner)
        other_labels.append(o.winner)
        meta.append({"ticket_id": j.ticket_id, "primary_margin": round(j.margin, 2)})
    return _compare(primary_labels, other_labels, meta)


# ------------------------------------------------------------------ human check
def make_human_worksheet(
    judgements: list[PairJudgement],
    tickets: dict[str, Ticket],
    candidates: dict[tuple[str, str], Candidate],
    out: Path,
    *,
    n: int = 20,
    seed: int = 1337,
) -> list[dict]:
    """Write a blind worksheet: responses shown as X/Y, judge's label withheld.

    The display order of X and Y is randomised per item and the mapping is stored
    in a separate answer key, so a reviewer cannot infer the label from position.
    """
    rng = random.Random(seed)
    sample = sample_for_check(judgements, n, seed)
    rows = []
    for j in sample:
        strategies = sorted(j.scores)
        rng.shuffle(strategies)
        t = tickets[j.ticket_id]
        rows.append(
            {
                "ticket_id": j.ticket_id,
                "category": t.category,
                "subject": t.subject,
                "body": t.body,
                "policy_ids": t.policy_ids,
                "X": candidates[(strategies[0], j.ticket_id)].text,
                "Y": candidates[(strategies[1], j.ticket_id)].text,
                "_key": {"X": strategies[0], "Y": strategies[1]},
                "_judge_winner": j.winner,
                "your_label": None,  # filled in as "X", "Y" or "tie"
            }
        )
    dump_json(out, rows)
    return rows


def score_human_worksheet(path: Path) -> AgreementResult:
    rows = json.loads(path.read_text(encoding="utf-8"))
    labelled = [r for r in rows if r.get("your_label") in {"X", "Y", "tie"}]
    if not labelled:
        raise ValueError(
            f"{path} has no filled-in 'your_label' fields - label them X / Y / tie first"
        )
    human = [
        r["_key"][r["your_label"]] if r["your_label"] != "tie" else "tie" for r in labelled
    ]
    judge = [r["_judge_winner"] for r in labelled]
    meta = [{"ticket_id": r["ticket_id"], "subject": r["subject"]} for r in labelled]
    return _compare(judge, human, meta)


def build_second_judge(model: str) -> Judge:
    from ..config import get_settings

    s = get_settings()
    return Judge(
        client=ChatClient(s.judge_base_url, s.judge_api_key, model, max_retries=s.judge_max_retries),
        model=model,
        temperature=0.0,
    )
