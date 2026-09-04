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


def make_guard_input_node(guards):
    """Screen and redact the inbound ticket before any model sees it."""

    def guard_input(state: TicketState) -> dict:
        t0 = time.perf_counter()
        body = state.get("body", "")
        d = guards.check_input(body)
        return {
            # The redacted body replaces the original for everything downstream, so a
            # card number never reaches the model, the logs, or the judge.
            "body": d.text,
            "guard_input_allowed": d.allowed,
            "guard_reasons": list(d.reasons),
            "guard_pii": dict(d.pii_found),
            "trace": [{
                "node": "guard_input", "latency_s": round(time.perf_counter() - t0, 4),
                "allowed": d.allowed, "pii": d.pii_found, "rail_s": d.rail_latency_s,
                "rail_available": d.rail_available,
            }],
        }

    return guard_input


def make_guard_output_node(guards):
    """Screen and redact the draft, and validate its structure, before it can be sent."""
    from ..guardrails.schema import validate_response

    def guard_output(state: TicketState) -> dict:
        t0 = time.perf_counter()
        draft = state.get("draft", "")
        d = guards.check_output(draft, user_text=state.get("body", ""))
        issues = validate_response(d.text)
        return {
            "draft": d.text,
            "guard_output_allowed": d.allowed,
            "guard_reasons": list(state.get("guard_reasons", [])) + list(d.reasons)
            + issues.reasons(),
            "schema_issues": issues.reasons(),
            "trace": [{
                "node": "guard_output", "latency_s": round(time.perf_counter() - t0, 4),
                "allowed": d.allowed, "pii": d.pii_found, "schema_ok": issues.ok,
                "rail_s": d.rail_latency_s,
            }],
        }

    return guard_output


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

        # A guardrail verdict is not advisory. An output the rails refused, or one that
        # fails structural validation, must never reach the customer regardless of how
        # confident the resolution model was.
        if state.get("guard_output_allowed") is False:
            escalate = True
            reasons.append("output guardrail refused the draft")
        if state.get("guard_input_allowed") is False:
            escalate = True
            reasons.append("input guardrail refused the ticket")
        for issue in state.get("schema_issues", []) or []:
            escalate = True
            reasons.append(f"schema: {issue}")

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
    tracer: Any = None

    def run(self, ticket_id: str, subject: str, body: str, **extra: Any) -> TicketState:
        t0 = time.perf_counter()
        state: TicketState = {
            "ticket_id": ticket_id,
            "subject": subject,
            "body": body,
            "trace": [],
            **extra,  # type: ignore[typeddict-item]
        }
        if self.tracer is None or not getattr(self.tracer, "enabled", False):
            out = self.graph.invoke(state)
        else:
            with self.tracer.ticket(ticket_id, subject, body) as tr:
                out = self.graph.invoke(state)
                # Roll the per-node trace up onto the root span, so opening one ticket
                # shows the routing decision and its reasons without drilling in.
                tr.update_ticket(
                    output={
                        "queue": out.get("queue"),
                        "final_response": out.get("final_response", "")[:2000],
                    },
                    metadata={
                        "category": out.get("category"),
                        "urgency": out.get("urgency"),
                        "escalated": out.get("escalate"),
                        "escalation_reasons": out.get("escalation_reasons", []),
                        "triage_source": out.get("triage_source"),
                        "guard_reasons": out.get("guard_reasons", []),
                        "pii_redacted": out.get("guard_pii", {}),
                        "resolution_model": out.get("resolution_model"),
                    },
                )
                tr.score("confidence", float(out.get("confidence", 0.0)),
                         "; ".join(out.get("escalation_reasons", [])[:2]))
                tr.score("auto_resolved", 0.0 if out.get("escalate") else 1.0)
            self.tracer.flush()

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
    guards=None,
    tracer=None,
) -> Pipeline:
    s = get_settings()
    thr = s.escalation_confidence_threshold if threshold is None else threshold

    if use_llm_triage and triage_client is None:
        triage_client, triage_model = build_triage_client()
    if not use_llm_triage:
        triage_client = None

    def traced(name: str, fn):
        """Wrap a node so each hop is its own typed Langfuse observation."""
        if tracer is None or not getattr(tracer, "enabled", False):
            return fn

        def wrapper(state: TicketState) -> dict:
            with tracer.node(name) as span:
                result = fn(state)
                entry = (result.get("trace") or [{}])[-1]
                if name == "resolution":
                    span.set_usage(
                        model=result.get("resolution_model", ""),
                        output_tokens=int(result.get("resolution_tokens", 0) or 0),
                    )
                span.finish(
                    output={k: v for k, v in result.items() if k != "trace"},
                    **{k: v for k, v in entry.items() if k != "node"},
                )
                return result

        return wrapper

    g = StateGraph(TicketState)
    if guards is not None:
        g.add_node("guard_input", traced("guard_input", make_guard_input_node(guards)))
    g.add_node("triage", traced("triage", make_triage_node(triage_client, triage_model)))
    g.add_node("retrieve", traced("retrieve", make_retrieve_node()))
    g.add_node("resolution", traced("resolution", make_resolution_node(resolver)))
    if guards is not None:
        g.add_node("guard_output", traced("guard_output", make_guard_output_node(guards)))
    g.add_node("gate", traced("gate", make_gate_node(thr)))
    g.add_node("respond", traced("respond", respond))
    g.add_node("escalate", traced("escalate", escalate))

    if guards is not None:
        g.set_entry_point("guard_input")
        g.add_edge("guard_input", "triage")
    else:
        g.set_entry_point("triage")
    g.add_edge("triage", "retrieve")
    g.add_edge("retrieve", "resolution")
    if guards is not None:
        g.add_edge("resolution", "guard_output")
        g.add_edge("guard_output", "gate")
    else:
        g.add_edge("resolution", "gate")
    g.add_conditional_edges("gate", _route, {"respond": "respond", "escalate": "escalate"})
    g.add_edge("respond", END)
    g.add_edge("escalate", END)

    return Pipeline(graph=g.compile(), threshold=thr, tracer=tracer)
