"""Resumability of the Phase 1 pipeline.

The judging stage is ~900 hosted-model calls. What is pinned here is that a re-run
does not pay for work already on disk, and that a run killed mid-write can still be
resumed — the two properties the checkpointing exists for.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sentineldesk.data.schema import Candidate, JudgeVerdict, Ticket, read_jsonl
from sentineldesk.prefs.build_pairs import _JudgementStore, _load_candidates, build_pairs
from sentineldesk.prefs.judge import PairJudgement, aggregate


def _ticket(i: int) -> Ticket:
    return Ticket(
        id=f"T{i:03d}", category="billing", urgency="low",
        subject="Refund", body="I was charged twice, can I get a refund?",
        policy_ids=["BIL-01"],
    )


def _judgement(ticket_id: str, winner: str = "grounded") -> PairJudgement:
    v1 = JudgeVerdict(
        ticket_id=ticket_id, first_shown="grounded", winner="A",
        scores={"A": {"correctness": 3, "completeness": 2, "conciseness": 2, "tone": 2},
                "B": {"correctness": 0, "completeness": 1, "conciseness": 1, "tone": 2}},
        rationale="A matches BIL-01",
    )
    v2 = JudgeVerdict(
        ticket_id=ticket_id, first_shown="rushed", winner="B",
        scores={"A": {"correctness": 0, "completeness": 1, "conciseness": 1, "tone": 2},
                "B": {"correctness": 3, "completeness": 2, "conciseness": 2, "tone": 2}},
        rationale="the other one matches BIL-01",
    )
    j = aggregate(ticket_id, "grounded", "rushed", v1, v2)
    assert j.winner == winner
    return j


# ------------------------------------------------------------------ aggregation
def test_agreeing_orders_produce_a_consistent_verdict():
    j = _judgement("T001")
    assert j.consistent
    assert j.winner == "grounded"
    assert j.loser == "rushed"


def test_disagreeing_orders_collapse_to_a_tie():
    """A verdict that flips when the order flips is position bias, not signal."""
    same = {"A": {"correctness": 3, "completeness": 2, "conciseness": 2, "tone": 2},
            "B": {"correctness": 1, "completeness": 1, "conciseness": 1, "tone": 1}}
    v1 = JudgeVerdict(ticket_id="T1", first_shown="grounded", winner="A", scores=same)
    v2 = JudgeVerdict(ticket_id="T1", first_shown="rushed", winner="A", scores=same)
    j = aggregate("T1", "grounded", "rushed", v1, v2)
    assert not j.consistent
    assert j.winner == "tie"
    assert j.loser is None


def test_scores_are_averaged_across_both_orders_per_strategy():
    j = _judgement("T001")
    # grounded scored 3 as A in order 1 and 3 as B in order 2
    assert j.scores["grounded"]["correctness"] == pytest.approx(3.0)
    assert j.scores["rushed"]["correctness"] == pytest.approx(0.0)


# ------------------------------------------------------------------ persistence
def test_judgement_store_round_trips_through_disk(tmp_path: Path):
    path = tmp_path / "judgements.jsonl"
    store = _JudgementStore(path)
    store.add(_judgement("T001"))
    store.add(_judgement("T002"))

    reloaded = _JudgementStore(path)
    assert reloaded.has("T001") and reloaded.has("T002")
    assert not reloaded.has("T999")
    got = {j.ticket_id: j for j in reloaded.all()}
    assert got["T001"].winner == "grounded"
    assert got["T001"].consistent
    assert got["T001"].verdicts[0].rationale == "A matches BIL-01"


def test_judgement_store_survives_a_torn_final_line(tmp_path: Path):
    """A run killed mid-write must still be resumable, not fail to load."""
    path = tmp_path / "judgements.jsonl"
    store = _JudgementStore(path)
    store.add(_judgement("T001"))
    with path.open("a") as fh:
        fh.write('{"ticket_id": "T002", "winner": "grou')  # killed mid-write

    reloaded = _JudgementStore(path)
    assert reloaded.has("T001")
    assert not reloaded.has("T002")
    assert len(reloaded.all()) == 1


def test_candidate_cache_is_keyed_by_strategy_and_ticket(tmp_path: Path):
    path = tmp_path / "candidates.jsonl"
    from sentineldesk.data.schema import write_jsonl

    write_jsonl(path, [
        Candidate(ticket_id="T001", strategy="grounded", text="a"),
        Candidate(ticket_id="T001", strategy="rushed", text="b"),
    ])
    cache = _load_candidates(path)
    assert cache[("grounded", "T001")].text == "a"
    assert cache[("rushed", "T001")].text == "b"
    assert _load_candidates(tmp_path / "missing.jsonl") == {}


# ------------------------------------------------------------------ end to end
class _CountingGenerator:
    def __init__(self) -> None:
        self.seen: list[int] = []

    def generate(self, batches, **kw):
        self.seen.append(len(batches))
        return ["A grounded reply citing the 30-day refund window." for _ in batches]


class _CountingJudge:
    model = "stub"

    def __init__(self) -> None:
        self.calls = 0

    def judge_stream(self, items, *, concurrency=4):
        for ticket, _x, _y in items:
            self.calls += 1
            yield _judgement(ticket.id)


def test_a_second_run_regenerates_and_rejudges_nothing(tmp_path: Path):
    tickets = [_ticket(i) for i in range(3)]

    gen1, judge1 = _CountingGenerator(), _CountingJudge()
    first = build_pairs(tickets, gen1, judge1, out_dir=tmp_path)
    assert judge1.calls == 3
    assert sum(gen1.seen) == 6  # 3 tickets x 2 strategies
    assert len(first.pairs) == 3

    gen2, judge2 = _CountingGenerator(), _CountingJudge()
    second = build_pairs(tickets, gen2, judge2, out_dir=tmp_path)
    assert judge2.calls == 0
    assert gen2.seen == []
    assert len(second.pairs) == 3


def test_new_tickets_added_later_are_the_only_ones_processed(tmp_path: Path):
    build_pairs([_ticket(0), _ticket(1)], _CountingGenerator(), _CountingJudge(), out_dir=tmp_path)

    gen, judge = _CountingGenerator(), _CountingJudge()
    grown = build_pairs(
        [_ticket(0), _ticket(1), _ticket(2)], gen, judge, out_dir=tmp_path
    )
    assert judge.calls == 1
    assert sum(gen.seen) == 2  # one new ticket, two strategies
    assert len(grown.pairs) == 3


def test_pairs_carry_the_canonical_prompt_and_the_right_sides(tmp_path: Path):
    result = build_pairs([_ticket(0)], _CountingGenerator(), _CountingJudge(), out_dir=tmp_path)
    pair = result.pairs[0]
    assert pair.chosen_strategy == "grounded"
    assert pair.rejected_strategy == "rushed"
    assert "POLICY EXCERPTS" in pair.prompt and "CUSTOMER TICKET" in pair.prompt
    # both sides trained against one shared prompt, as DPO requires
    assert pair.prompt.count("CUSTOMER TICKET") == 1
    assert read_jsonl(tmp_path / "pairs.jsonl", type(pair)) if (tmp_path / "pairs.jsonl").exists() else True


def test_inconsistent_judgements_never_become_training_pairs(tmp_path: Path):
    class _FlipFlopJudge(_CountingJudge):
        def judge_stream(self, items, *, concurrency=4):
            same = {"A": {"correctness": 3, "completeness": 2, "conciseness": 2, "tone": 2},
                    "B": {"correctness": 1, "completeness": 1, "conciseness": 1, "tone": 1}}
            for ticket, _x, _y in items:
                v1 = JudgeVerdict(ticket_id=ticket.id, first_shown="grounded", winner="A", scores=same)
                v2 = JudgeVerdict(ticket_id=ticket.id, first_shown="rushed", winner="A", scores=same)
                yield aggregate(ticket.id, "grounded", "rushed", v1, v2)

    result = build_pairs([_ticket(0)], _CountingGenerator(), _FlipFlopJudge(), out_dir=tmp_path)
    assert result.judgements  # the verdict is recorded
    assert result.pairs == []  # but it does not train anything
    assert result.stats["order_inconsistency_rate"] == 1.0


# ------------------------------------------------------------------ length balancing
def _lp(chosen_len: int, rejected_len: int, i: int = 0):
    from sentineldesk.data.schema import PreferencePair

    return PreferencePair(
        ticket_id=f"T{i:03d}", prompt="p", chosen="c" * chosen_len, rejected="r" * rejected_len,
        chosen_strategy="grounded", rejected_strategy="rushed",
    )


def test_balance_length_leaves_an_already_matched_set_alone():
    from sentineldesk.prefs.build_pairs import balance_length

    pairs = [_lp(500, 500, i) for i in range(20)]
    kept, stats = balance_length(pairs)
    assert len(kept) == 20
    assert stats["pairs_dropped_for_length"] == 0
    assert not stats["length_balancing_applied"]


def test_balance_length_pulls_a_short_chosen_skew_toward_parity():
    """The v1 failure: chosen systematically shorter than rejected."""
    from sentineldesk.prefs.build_pairs import balance_length

    pairs = [_lp(300, 900, i) for i in range(6)] + [_lp(600, 600, i) for i in range(6, 24)]
    kept, stats = balance_length(pairs)
    assert stats["length_ratio_before_balancing"] < 0.93
    assert 0.93 <= stats["length_ratio_after_balancing"] <= 1.07
    assert stats["pairs_dropped_for_length"] > 0
    assert len(kept) < len(pairs)


def test_balance_length_pulls_a_long_chosen_skew_toward_parity():
    from sentineldesk.prefs.build_pairs import balance_length

    pairs = [_lp(900, 300, i) for i in range(6)] + [_lp(600, 600, i) for i in range(6, 24)]
    _, stats = balance_length(pairs)
    assert stats["length_ratio_before_balancing"] > 1.07
    assert 0.93 <= stats["length_ratio_after_balancing"] <= 1.07


def test_balance_length_respects_the_drop_cap_and_says_so():
    """A matched dataset that discarded most of its signal is not an improvement."""
    from sentineldesk.prefs.build_pairs import balance_length

    pairs = [_lp(100, 900, i) for i in range(20)]  # hopelessly skewed, every pair
    kept, stats = balance_length(pairs, max_drop_frac=0.25)
    assert stats["pairs_dropped_for_length"] == 5
    assert stats["hit_drop_cap"] is True
    assert len(kept) == 15
