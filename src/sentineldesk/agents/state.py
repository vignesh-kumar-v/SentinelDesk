"""The state that flows through the LangGraph pipeline."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict


def _append(left: list, right: list) -> list:
    """Reducer so each node appends its own trace entry instead of overwriting."""
    return (left or []) + (right or [])


class TicketState(TypedDict, total=False):
    # input
    ticket_id: str
    subject: str
    body: str
    true_category: str  # only present when replaying labelled tickets, for scoring

    # triage node
    category: str
    urgency: str
    triage_confidence: float
    triage_source: str  # "llm" | "keyword-fallback"

    # retrieve node
    policy_ids: list[str]

    # resolution node
    draft: str
    resolution_model: str
    resolution_latency_s: float
    resolution_tokens: int
    mean_logprob: float | None

    # confidence node
    confidence: float
    confidence_signals: dict[str, Any]

    # escalation
    escalate: bool
    escalation_reasons: list[str]
    queue: str

    # output
    final_response: str
    trace: Annotated[list[dict], _append]
