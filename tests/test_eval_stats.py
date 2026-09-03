"""Statistics checked against values computable by hand or from published tables."""

from __future__ import annotations

import math

import pytest

from sentineldesk.eval.arena import ARM_BASE, ARM_TUNED, ArenaOutcome, summarise_arena
from sentineldesk.eval.stats import pearson, two_sided_binomial_p, wilson_interval
from sentineldesk.prefs.spotcheck import cohens_kappa


def test_wilson_matches_published_value():
    lo, hi = wilson_interval(60, 100)
    assert lo == pytest.approx(0.5018, abs=5e-4)
    assert hi == pytest.approx(0.6906, abs=5e-4)


def test_wilson_stays_inside_bounds_at_the_extremes():
    """The reason for Wilson over the normal approximation."""
    lo, hi = wilson_interval(0, 20)
    assert lo == 0.0 and 0.0 < hi < 0.2
    lo, hi = wilson_interval(20, 20)
    assert hi == 1.0 and 0.8 < lo < 1.0


def test_wilson_narrows_as_n_grows():
    widths = [hi - lo for hi, lo in ((wilson_interval(n // 2, n)[::-1]) for n in (20, 100, 1000))]
    assert widths == sorted(widths, reverse=True)


def test_binomial_p_exact_values():
    assert two_sided_binomial_p(50, 100) == pytest.approx(1.0)
    # both tails of a 10-flip run: 2 * 0.5^10
    assert two_sided_binomial_p(10, 10) == pytest.approx(2 * 0.5**10)
    assert two_sided_binomial_p(60, 100) == pytest.approx(0.0569, abs=1e-3)


def test_binomial_p_is_symmetric():
    assert two_sided_binomial_p(30, 100) == pytest.approx(two_sided_binomial_p(70, 100))


def test_pearson_endpoints():
    assert pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert pearson([1, 2, 3], [6, 4, 2]) == pytest.approx(-1.0)
    assert pearson([1, 1, 1], [1, 2, 3]) == 0.0


def test_kappa_is_zero_at_chance_and_one_at_perfect():
    assert cohens_kappa(["a", "b"] * 10, ["a", "b"] * 10) == pytest.approx(1.0)
    assert math.isnan(cohens_kappa(["a"] * 10, ["a"] * 10))


def test_kappa_punishes_a_constant_labeller_that_raw_agreement_would_reward():
    """The reason kappa is reported instead of raw agreement."""
    judge = ["grounded"] * 17 + ["rushed"] * 3
    lazy = ["grounded"] * 20
    raw = sum(1 for a, b in zip(judge, lazy, strict=True) if a == b) / 20
    assert raw == 0.85
    assert cohens_kappa(judge, lazy) == pytest.approx(0.0, abs=1e-9)


def _outcome(winner, cat="billing", len_t=100, len_b=100, consistent=True):
    scores = {
        ARM_TUNED: {"correctness": 2, "completeness": 1, "conciseness": 1, "tone": 1},
        ARM_BASE: {"correctness": 1, "completeness": 1, "conciseness": 1, "tone": 1},
    }
    return ArenaOutcome(
        ticket_id="t", category=cat, winner=winner, consistent=consistent, margin=1.0,
        scores=scores, lengths={ARM_TUNED: len_t, ARM_BASE: len_b}, shown_first=ARM_TUNED,
    )


def test_arena_reports_both_win_rates_and_they_differ_when_there_are_ties():
    outcomes = [_outcome(ARM_TUNED)] * 6 + [_outcome(ARM_BASE)] * 2 + [_outcome("tie")] * 2
    s = summarise_arena(outcomes)
    assert s["wins_tuned"] == 6 and s["losses_tuned"] == 2 and s["ties"] == 2
    assert s["win_rate_decisive"] == pytest.approx(0.75)   # ties dropped
    assert s["win_rate_adjusted"] == pytest.approx(0.70)   # ties counted as half
    assert s["win_rate_adjusted"] < s["win_rate_decisive"]


def test_arena_flags_length_correlation_when_short_answers_win():
    """A tuned arm that wins only when it is shorter is the reward-hacking signature."""
    outcomes = [_outcome(ARM_TUNED, len_t=50, len_b=200) for _ in range(8)]
    outcomes += [_outcome(ARM_BASE, len_t=250, len_b=200) for _ in range(8)]
    s = summarise_arena(outcomes)
    assert s["corr_win_vs_length_delta"] < -0.9


def test_arena_reports_order_inconsistency():
    outcomes = [_outcome(ARM_TUNED)] * 8 + [_outcome("tie", consistent=False)] * 2
    assert summarise_arena(outcomes)["order_inconsistency_rate"] == pytest.approx(0.2)


def test_arena_handles_an_empty_run():
    assert summarise_arena([])["n"] == 0
