"""Triage: classify category and urgency.

An LLM call with a deterministic keyword classifier behind it. The fallback is not
decoration - the triage model is a separate service from the resolution model, and
a graph whose first node hard-fails when that service blips routes nothing at all.
Falling back to keywords degrades accuracy but keeps the pipeline answering, and
the state records which path ran so the degradation is visible rather than silent.
"""

from __future__ import annotations

import re
import time

from ..config import get_settings
from ..data.schema import CATEGORIES
from ..llm import ChatClient, LLMError
from ..logging_utils import get_logger
from .state import TicketState

log = get_logger(__name__)

TRIAGE_SYSTEM = "You classify inbound support tickets. You output only JSON."

TRIAGE_PROMPT = """Classify this support ticket.

category - exactly one of:
  billing         - charges, refunds, invoices, plans, proration, failed payments
  technical       - errors, sync, exports, API, platform support, outages
  account_access  - passwords, 2FA, lockouts, seats, roles, account deletion
  shipping        - delivery, tracking, lost parcels, returns, addresses
  product_info    - what a plan includes, trials, compliance, roadmap

urgency - one of low, medium, high. high means the customer is blocked from
working, money is at stake right now, or they are threatening to leave.

confidence - 0.0 to 1.0, how sure you are of the category.

TICKET
Subject: {subject}
{body}

Return only: {{"category": "...", "urgency": "...", "confidence": 0.0}}"""

# Ordered most-specific first: "refund on my order" is billing, not shipping, and a
# dict-order-independent scorer would call that a tie.
_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("account_access", ("password", "2fa", "two-factor", "locked out", "recovery code",
                        "sign in", "log in", "seat", "role", "owner", "admin", "delete my account")),
    ("billing", ("refund", "charge", "charged", "invoice", "billing", "payment", "card",
                 "proration", "prorate", "upgrade", "downgrade", "subscription", "vat", "receipt")),
    ("shipping", ("shipping", "delivery", "parcel", "package", "tracking", "courier",
                  "rma", "return label", "shipped", "address")),
    ("technical", ("error", "sync-409", "sync", "export", "api", "rate limit", "429",
                   "crash", "bug", "outage", "not working", "linux", "install")),
    ("product_info", ("plan", "trial", "sso", "saml", "roadmap", "soc 2", "hipaa",
                      "data residency", "feature", "pricing")),
]

_URGENT = ("urgent", "asap", "immediately", "critical", "blocked", "cancel my", "churn",
           "escalate", "lawyer", "unacceptable", "right now", "deadline")


def keyword_triage(subject: str, body: str) -> tuple[str, str, float]:
    text = f"{subject} {body}".lower()
    scores = {
        cat: sum(1 for kw in kws if kw in text) for cat, kws in _KEYWORDS
    }
    # Urgency is independent of category and must be decided before any early
    # return: a ticket that matches no category keyword can still be on fire, and
    # routing it at default urgency because the classifier shrugged is exactly the
    # ticket a support queue cannot afford to bury.
    urgency = "high" if any(u in text for u in _URGENT) else "medium"

    best = max(scores, key=lambda c: scores[c])
    hits = scores[best]
    if hits == 0:
        return "technical", urgency, 0.2

    runner_up = sorted(scores.values(), reverse=True)[1]
    # Confidence from the margin over the next-best category, not the raw hit count:
    # a ticket matching four billing words and four shipping words is ambiguous no
    # matter how many words it matched.
    confidence = min(0.35 + 0.15 * (hits - runner_up), 0.75)
    return best, urgency, max(confidence, 0.25)


def make_triage_node(client: ChatClient | None, model: str):
    def triage(state: TicketState) -> dict:
        t0 = time.perf_counter()
        subject, body = state.get("subject", ""), state.get("body", "")
        source = "llm"
        category = urgency = None
        confidence = 0.0

        if client is not None:
            try:
                obj, _ = client.chat_json(
                    [
                        {"role": "system", "content": TRIAGE_SYSTEM},
                        {
                            "role": "user",
                            "content": TRIAGE_PROMPT.format(subject=subject, body=body),
                        },
                    ],
                    required_keys=("category",),
                    model=model,
                    temperature=0.0,
                    max_tokens=2000,
                )
                cand = str(obj.get("category", "")).strip().lower()
                if cand in CATEGORIES:
                    category = cand
                    urgency = str(obj.get("urgency", "medium")).strip().lower()
                    confidence = float(obj.get("confidence", 0.7) or 0.7)
            except (LLMError, ValueError, TypeError) as exc:
                log.warning("triage llm failed (%s); falling back to keywords", type(exc).__name__)

        if category is None:
            category, urgency, confidence = keyword_triage(subject, body)
            source = "keyword-fallback"
        if urgency not in {"low", "medium", "high"}:
            urgency = "medium"

        return {
            "category": category,
            "urgency": urgency,
            "triage_confidence": max(0.0, min(confidence, 1.0)),
            "triage_source": source,
            "trace": [
                {
                    "node": "triage",
                    "latency_s": round(time.perf_counter() - t0, 4),
                    "category": category,
                    "urgency": urgency,
                    "source": source,
                }
            ],
        }

    return triage


def build_triage_client() -> tuple[ChatClient | None, str]:
    s = get_settings()
    try:
        return ChatClient(s.judge_base_url, s.judge_api_key, s.judge_model, max_retries=2), s.judge_model
    except Exception as exc:  # noqa: BLE001
        log.warning("no triage client (%s); keyword triage only", exc)
        return None, "keyword-fallback"


_WORD = re.compile(r"[a-z0-9]+")
