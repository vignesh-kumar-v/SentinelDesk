"""Triage, confidence scoring and graph routing.

A stub resolver keeps these offline and deterministic. What is being pinned is the
routing contract: which inputs must reach a human, which must not, and that a
degraded triage path stays visible in the state rather than passing silently.
"""

from __future__ import annotations

import pytest

from sentineldesk.agents.confidence import (
    mandatory_escalation_hits,
    policy_overlap,
    repetition_ratio,
    score_draft,
)
from sentineldesk.agents.graph import ESCALATION_QUEUE, build_pipeline
from sentineldesk.agents.triage import keyword_triage

GOOD_BILLING = (
    "Refunds are available within 30 days of the charge date, and approved refunds land "
    "in 5-7 business days on the original payment method. I have started that for you."
)


class StubResolver:
    name = "stub"

    def __init__(self, text: str = GOOD_BILLING, raises: bool = False) -> None:
        self.text = text
        self.raises = raises
        self.calls = 0

    def resolve(self, messages):
        self.calls += 1
        if self.raises:
            raise RuntimeError("backend exploded")
        return self.text, {"model": "stub", "latency_s": 0.01, "tokens": 40, "mean_logprob": None}


def _pipe(resolver, **kw):
    return build_pipeline(resolver, use_llm_triage=False, **kw)


# ------------------------------------------------------------------ triage
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I was charged twice for my subscription, please refund the duplicate", "billing"),
        ("I lost my recovery code and cannot get past two-factor", "account_access"),
        ("My parcel has no tracking update and delivery was due last week", "shipping"),
        ("Getting SYNC-409 every time the app syncs", "technical"),
        ("Does the Pro plan include SSO or is that Enterprise only", "product_info"),
    ],
)
def test_keyword_triage_finds_the_obvious_category(text, expected):
    assert keyword_triage("", text)[0] == expected


def test_keyword_triage_confidence_reflects_the_margin_not_the_hit_count():
    """Many hits split evenly across categories is ambiguous, however many there are."""
    _, _, clear = keyword_triage("", "refund refund invoice invoice billing charge")
    _, _, muddy = keyword_triage("", "refund invoice shipping parcel tracking charge")
    assert clear > muddy


def test_keyword_triage_falls_back_with_low_confidence_when_nothing_matches():
    _, _, conf = keyword_triage("", "hello I have a general question about things")
    assert conf <= 0.25


def test_keyword_triage_detects_urgency():
    assert keyword_triage("", "this is urgent, we are blocked")[1] == "high"
    assert keyword_triage("", "just a small question about my invoice")[1] != "high"


# ------------------------------------------------------------------ confidence
def test_repetition_ratio_flags_a_looping_generation():
    assert repetition_ratio("the cat sat on the mat today " * 6) > 0.5
    assert repetition_ratio("a completely unique sentence with no repeated phrases at all") == 0.0


def test_policy_overlap_separates_grounded_from_invented():
    grounded = policy_overlap(GOOD_BILLING, ["BIL-01"])
    invented = policy_overlap("Sure, I have wired the money to your crypto wallet.", ["BIL-01"])
    assert grounded > 0.4
    assert invented < 0.15


def test_confidence_is_high_for_a_grounded_specific_draft():
    r = score_draft(GOOD_BILLING, policy_ids=["BIL-01"], triage_confidence=0.9)
    assert r.score > 0.8
    assert r.reasons == []


def test_confidence_collapses_for_a_hedged_ungrounded_draft():
    r = score_draft(
        "Hmm, I think maybe you should please contact support about this.",
        policy_ids=["BIL-01"], triage_confidence=0.9,
    )
    assert r.score < 0.4
    assert any("hedged" in x for x in r.reasons)


def test_low_triage_confidence_drags_the_score_down():
    """Wrong retrieved policies make the groundedness signal measure the wrong target."""
    high = score_draft(GOOD_BILLING, policy_ids=["BIL-01"], triage_confidence=0.9).score
    low = score_draft(GOOD_BILLING, policy_ids=["BIL-01"], triage_confidence=0.2).score
    assert low < high


def test_mandatory_escalation_only_fires_when_the_policy_is_in_context():
    text = "I lost my recovery codes, please disable 2FA"
    assert mandatory_escalation_hits("", text, ["ACC-02"]) == ["ACC-02"]
    assert mandatory_escalation_hits("", text, ["BIL-01"]) == []


# ------------------------------------------------------------------ graph
def test_a_confident_grounded_draft_is_returned_to_the_customer():
    out = _pipe(StubResolver()).run("T1", "Refund question", "I was charged twice, can I get a refund?")
    assert out["queue"] == "auto-resolved"
    assert out["final_response"] == GOOD_BILLING
    assert not out["escalate"]


def test_the_graph_visits_every_node_in_order():
    out = _pipe(StubResolver()).run("T1", "Refund", "I was charged twice, refund please")
    assert [t["node"] for t in out["trace"]] == [
        "triage", "retrieve", "resolution", "gate", "respond", "_total"
    ]


def test_a_weak_draft_escalates_with_stated_reasons():
    out = _pipe(StubResolver("umm not sure")).run("T2", "Refund", "I was charged twice, refund?")
    assert out["queue"] == ESCALATION_QUEUE
    assert out["escalation_reasons"]


def test_a_flagged_request_escalates_even_when_the_draft_looks_confident():
    """The dangerous case: fluent, high-confidence, and forbidden by policy."""
    confident = (
        "Two-factor lockouts are handled by support. Recovery codes are the documented "
        "path and identity verification through the recovery form is required. "
        "I have disabled two-factor on your account now."
    )
    out = _pipe(StubResolver(confident)).run(
        "T3", "2FA lockout", "I lost my recovery codes, please disable 2FA for me"
    )
    assert out["queue"] == ESCALATION_QUEUE
    assert any("ACC-02" in r for r in out["escalation_reasons"])


def test_escalation_withholds_the_draft_from_the_customer():
    draft = "Sure, I have disabled two-factor for you, no verification needed."
    out = _pipe(StubResolver(draft)).run(
        "T4", "2FA", "I lost my recovery codes, please disable 2FA"
    )
    assert draft not in out["final_response"]
    assert out["draft"] == draft  # still attached for the human agent


def test_a_dead_resolution_backend_escalates_instead_of_crashing():
    out = _pipe(StubResolver(raises=True)).run("T5", "Refund", "I was charged twice, refund?")
    assert out["queue"] == ESCALATION_QUEUE
    assert any("no draft" in r for r in out["escalation_reasons"])


def test_threshold_controls_the_routing_decision():
    # A draft with one mild penalty, so it sits strictly between the two thresholds
    # rather than at a perfect 1.0 where no threshold below 1.0 could escalate it.
    resolver = StubResolver(
        "I think the refund window is 30 days from the charge date, and approved "
        "refunds land in 5-7 business days on the original payment method."
    )
    lenient = _pipe(resolver, threshold=0.1).run("T6", "Refund", "charged twice, refund?")
    strict = _pipe(resolver, threshold=0.95).run("T6", "Refund", "charged twice, refund?")
    assert 0.1 < lenient["confidence"] < 0.95
    assert lenient["queue"] == "auto-resolved"
    assert strict["queue"] == ESCALATION_QUEUE


def test_triage_source_is_recorded_so_a_degraded_path_is_visible():
    out = _pipe(StubResolver()).run("T7", "Refund", "charged twice, refund please")
    assert out["triage_source"] == "keyword-fallback"
