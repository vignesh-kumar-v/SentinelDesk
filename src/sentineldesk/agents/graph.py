"""The LangGraph pipeline: triage -> retrieve -> resolution -> gate -> respond|escalate."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, StateGraph

from ..config import get_settings
from ..logging_utils import get_logger
from .confidence import mandatory_escalation_hits, score_draft
from .resolution import Resolver, make_resolution_node, make_retrieve_node
from .state import TicketState
from .triage import build_triage_client, make_triage_node

log = get_logger(__name__)

ESCALATION_QUEUE = "human-tier-2"


def make_gate_node(threshold: float):
    """Score the draft and decide the route. The only node that can escalate."""

    def gate(state: TicketState) -> dict:
        t0 = time.perf_counter()
        draft = state.get("draft", "")
        report = score_draft(
            draft,
            policy_ids=state.get("policy_ids", []),
            triage_confidence=float(state.get("triage_confidence", 0.5)),
            mean_logprob=state.get("mean_logprob"),
        )
        reasons = list(report.reasons)
        escalate = report.score < threshold

        if not draft.strip():
            escalate = True
            reasons.append("resolution produced no draft")

        forced = mandatory_escalation_hits(
            state.get("subject", ""), state.get("body", ""), state.get("policy_ids", [])
        )
        if forced:
            # Deliberately independent of the confidence score. On these requests a
            # fluent, confident draft is the dangerous outcome, not a reassuring one.
            escalate = True
            reasons.append(f"policy requires human handling: {', '.join(forced)}")

        return {
            "confidence": round(report.score, 4),
            "confidence_signals": dict(report.signals) | {"forced_escalation": forced},
            "escalate": escalate,
            "escalation_reasons": reasons if escalate else [],
            "trace": [
                {
                    "node": "gate",
                    "latency_s": round(time.perf_counter() - t0, 4),
                    "confidence": round(report.score, 4),
                    "threshold": threshold,
                    "escalate": escalate,
                }
            ],
        }

    return gate


def respond(state: TicketState) -> dict:
    return {
        "final_response": state.get("draft", ""),
        "queue": "auto-resolved",
        "trace": [{"node": "respond", "latency_s": 0.0}],
    }


def escalate(state: TicketState) -> dict:
    reasons = state.get("escalation_reasons", [])
    summary = (
        f"Escalated to {ESCALATION_QUEUE}. "
        f"Category: {state.get('category')}, urgency: {state.get('urgency')}. "
        f"Reasons: {'; '.join(reasons) if reasons else 'below confidence threshold'}."
    )
    return {
        # The draft is deliberately not sent to the customer. It is attached for the
        # human agent, who is better served by seeing what the model produced than by
        # starting from a blank box.
        "final_response": summary,
        "queue": ESCALATION_QUEUE,
        "trace": [{"node": "escalation", "latency_s": 0.0, "queue": ESCALATION_QUEUE}],
    }


def _route(state: TicketState) -> str:
    return "escalate" if state.get("escalate") else "respond"


@dataclass
class Pipeline:
    graph: Any
    threshold: float

    def run(self, ticket_id: str, subject: str, body: str, **extra: Any) -> TicketState:
        t0 = time.perf_counter()
        state: TicketState = {
            "ticket_id": ticket_id,
            "subject": subject,
            "body": body,
            "trace": [],
            **extra,  # type: ignore[typeddict-item]
        }
        out = self.graph.invoke(state)
        out["trace"] = list(out.get("trace", [])) + [
            {"node": "_total", "latency_s": round(time.perf_counter() - t0, 4)}
        ]
        return out


def build_pipeline(
    resolver: Resolver,
    *,
    threshold: float | None = None,
    triage_client=None,
    triage_model: str = "",
    use_llm_triage: bool = True,
) -> Pipeline:
    s = get_settings()
    thr = s.escalation_confidence_threshold if threshold is None else threshold

    if use_llm_triage and triage_client is None:
        triage_client, triage_model = build_triage_client()
    if not use_llm_triage:
        triage_client = None

    g = StateGraph(TicketState)
    g.add_node("triage", make_triage_node(triage_client, triage_model))
    g.add_node("retrieve", make_retrieve_node())
    g.add_node("resolution", make_resolution_node(resolver))
    g.add_node("gate", make_gate_node(thr))
    g.add_node("respond", respond)
    g.add_node("escalate", escalate)

    g.set_entry_point("triage")
    g.add_edge("triage", "retrieve")
    g.add_edge("retrieve", "resolution")
    g.add_edge("resolution", "gate")
    g.add_conditional_edges("gate", _route, {"respond": "respond", "escalate": "escalate"})
    g.add_edge("respond", END)
    g.add_edge("escalate", END)

    return Pipeline(graph=g.compile(), threshold=thr)
