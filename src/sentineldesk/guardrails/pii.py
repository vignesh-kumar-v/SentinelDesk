"""Deterministic PII detection and redaction.

Regex, not a model. PII redaction is one of the few jobs in an LLM pipeline where a
deterministic checker is strictly better: it is auditable, it costs microseconds, it
cannot be prompt-injected, and its recall on structured identifiers (cards, IBANs,
SSNs) is far above what an LLM will give you reliably. The LLM-based rails in
`topics.py` handle the judgement calls; this handles the ones with an answer.

Card numbers are additionally Luhn-checked, because a 16-digit order reference is not
a card number and redacting it would destroy information a support agent needs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PIIPattern:
    label: str
    pattern: re.Pattern[str]
    validator: str | None = None


def _luhn(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


PATTERNS: tuple[PIIPattern, ...] = (
    PIIPattern("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b")),
    PIIPattern(
        "CARD",
        re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
        validator="luhn",
    ),
    PIIPattern("IBAN", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")),
    PIIPattern("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    # Deliberately narrow: a bare 10-digit run is more often an order id than a phone
    # number, and over-redacting the ticket body makes the agent worse at its job.
    PIIPattern(
        "PHONE",
        re.compile(r"(?<![\w-])(?:\+\d{1,3}[ -]?)?(?:\(\d{3}\)|\d{3})[ -]\d{3}[ -]\d{4}(?![\w-])"),
    ),
    PIIPattern("IPV4", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    PIIPattern(
        "API_KEY",
        # The body must allow internal - and _ separators. Without them this missed
        # "sk-live-..." (the literal Stripe format) and "api_key_...", which is how the
        # adversarial set caught it. The leading [A-Za-z0-9] and the 15-char tail keep
        # "key-value pairs" and "sk-short" out.
        re.compile(r"\b(?:sk|pk|api|key|token)[-_][A-Za-z0-9][A-Za-z0-9_-]{14,}\b", re.IGNORECASE),
    ),
)


@dataclass
class PIIResult:
    text: str
    found: dict[str, int] = field(default_factory=dict)

    @property
    def redacted(self) -> bool:
        return bool(self.found)

    @property
    def labels(self) -> list[str]:
        return sorted(self.found)


def _valid(p: PIIPattern, match: str) -> bool:
    if p.validator == "luhn":
        digits = re.sub(r"\D", "", match)
        return 13 <= len(digits) <= 19 and _luhn(digits)
    return True


def scan(text: str) -> dict[str, int]:
    """Count PII occurrences by label without modifying the text."""
    found: dict[str, int] = {}
    for p in PATTERNS:
        hits = [m.group(0) for m in p.pattern.finditer(text) if _valid(p, m.group(0))]
        if hits:
            found[p.label] = len(hits)
    return found


def redact(text: str) -> PIIResult:
    """Replace every detected identifier with a typed placeholder.

    Typed rather than a blanket [REDACTED] so a human agent reading the escalation can
    still tell that a card number was present without seeing it.
    """
    found: dict[str, int] = {}
    out = text
    for p in PATTERNS:

        def sub(m: re.Match[str], _p: PIIPattern = p) -> str:
            if not _valid(_p, m.group(0)):
                return m.group(0)
            found[_p.label] = found.get(_p.label, 0) + 1
            return f"[{_p.label}_REDACTED]"

        out = p.pattern.sub(sub, out)
    return PIIResult(text=out, found=found)
