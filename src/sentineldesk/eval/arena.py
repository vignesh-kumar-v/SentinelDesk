"""Phase 5: blind head-to-head between the DPO-tuned agent and the base-prompted one.

Fairness rules this implements, each of which is a way the number could otherwise be
quietly inflated:

* Both arms get the identical system prompt and the identical retrieved policies.
  The only difference between them is the weights. A baseline handicapped by a worse
  prompt would turn a prompt-engineering win into a claimed fine-tuning win.
* Held-out tickets only. These were split off before any preference pair was built,
  so nothing here was seen during Phase 1 or Phase 2.
* Every comparison is judged in both display orders, and a decision only counts as a
  win if the judge agrees with itself across the swap. Order-flips are reported as
  ties, not silently resolved.
* The judge is never told which arm is which, and the arm-to-slot assignment is
  randomised per ticket.
* Same rubric as Phase 1. Scoring the benchmark with a revised rubric would measure
  the rubric change rather than the fine-tune.
"""

from __future__ import annotations

import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from ..data.schema import Candidate, Ticket
from ..logging_utils import get_logger
from ..prefs.judge import Judge
from .stats import pearson, two_sided_binomial_p, wilson_interval

log = get_logger(__name__)

ARM_TUNED = "dpo_tuned"
ARM_BASE = "base_prompted"


@dataclass
class ArenaOutcome:
    ticket_id: str
    category: str
    winner: str  # ARM_TUNED | ARM_BASE | "tie"
    consistent: bool
    margin: float
    scores: dict[str, dict[str, float]]
    lengths: dict[str, int]
    shown_first: str


@dataclass
class ArenaResult:
    outcomes: list[ArenaOutcome] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def run_arena(
    tickets: list[Ticket],
    tuned: dict[str, str],
    base: dict[str, str],
    judge: Judge,
    *,
    seed: int = 1337,
    concurrency: int = 4,
) -> ArenaResult:
    """tuned/base map ticket_id -> response text."""
    rng = random.Random(seed)
    items, order = [], []
    for t in tickets:
        if t.id not in tuned or t.id not in base:
            continue
        # Randomise which arm occupies the first slot, on top of the judge's own
        # both-orders swap. The swap removes the judge's position bias; this removes
        # any systematic pairing between arm and slot in the aggregate.
        first = ARM_TUNED if rng.random() < 0.5 else ARM_BASE
        x_arm, y_arm = (first, ARM_BASE if first == ARM_TUNED else ARM_TUNED)
        texts = {ARM_TUNED: tuned[t.id], ARM_BASE: base[t.id]}
        items.append(
            (
                t,
                Candidate(ticket_id=t.id, strategy=x_arm, text=texts[x_arm], n_chars=len(texts[x_arm])),
                Candidate(ticket_id=t.id, strategy=y_arm, text=texts[y_arm], n_chars=len(texts[y_arm])),
            )
        )
        order.append((t, first))

    log.info("arena: %d held-out tickets x 2 orders = %d judge calls", len(items), 2 * len(items))
    judgements = judge.judge_many(items, concurrency=concurrency)

    outcomes = []
    for (t, first), j in zip(order, judgements, strict=True):
        if j is None:
            continue
        outcomes.append(
            ArenaOutcome(
                ticket_id=t.id,
                category=t.category,
                winner=j.winner,
                consistent=j.consistent,
                margin=j.margin,
                scores=j.scores,
                lengths={ARM_TUNED: len(tuned[t.id]), ARM_BASE: len(base[t.id])},
                shown_first=first,
            )
        )
    return ArenaResult(outcomes=outcomes, summary=summarise_arena(outcomes))


def summarise_arena(outcomes: list[ArenaOutcome]) -> dict:
    n = len(outcomes)
    if n == 0:
        return {"n": 0}

    wins = sum(1 for o in outcomes if o.winner == ARM_TUNED)
    losses = sum(1 for o in outcomes if o.winner == ARM_BASE)
    ties = n - wins - losses

    decisive = wins + losses
    # Two win-rates, because they answer different questions and quoting only the
    # flattering one is the standard way this metric gets abused. The decisive rate
    # ignores ties; the tie-adjusted rate counts each tie as half a win and is the
    # one to lead with, since a fine-tune that produces more ties has not improved
    # anything.
    wr_decisive = wins / decisive if decisive else 0.0
    wr_adjusted = (wins + 0.5 * ties) / n

    lo_d, hi_d = wilson_interval(wins, decisive)
    lo_a, hi_a = wilson_interval(wins + 0.5 * ties, n)

    by_cat: dict[str, dict] = {}
    grouped: dict[str, list[ArenaOutcome]] = defaultdict(list)
    for o in outcomes:
        grouped[o.category].append(o)
    for cat, group in sorted(grouped.items()):
        w = sum(1 for o in group if o.winner == ARM_TUNED)
        t_ = sum(1 for o in group if o.winner == "tie")
        by_cat[cat] = {
            "n": len(group),
            "wins": w,
            "ties": t_,
            "losses": len(group) - w - t_,
            "win_rate_adjusted": round((w + 0.5 * t_) / len(group), 4),
        }

    dims = ("correctness", "completeness", "conciseness", "tone")
    mean_scores = {
        arm: {
            d: round(statistics.mean(o.scores[arm][d] for o in outcomes if arm in o.scores), 3)
            for d in dims
        }
        for arm in (ARM_TUNED, ARM_BASE)
    }
    for arm in mean_scores:
        mean_scores[arm]["total"] = round(sum(mean_scores[arm][d] for d in dims), 3)

    len_tuned = [o.lengths[ARM_TUNED] for o in outcomes]
    len_base = [o.lengths[ARM_BASE] for o in outcomes]
    # The reward-hacking check the blueprint calls out by name: if the tuned model
    # wins mainly by being shorter, this correlation is where it shows up.
    length_delta = [t_ - b for t_, b in zip(len_tuned, len_base, strict=True)]
    won = [1.0 if o.winner == ARM_TUNED else 0.0 for o in outcomes]

    return {
        "n": n,
        "wins_tuned": wins,
        "losses_tuned": losses,
        "ties": ties,
        "win_rate_adjusted": round(wr_adjusted, 4),
        "win_rate_adjusted_ci95": [round(lo_a, 4), round(hi_a, 4)],
        "win_rate_decisive": round(wr_decisive, 4),
        "win_rate_decisive_ci95": [round(lo_d, 4), round(hi_d, 4)],
        "binomial_p_vs_50pct": round(two_sided_binomial_p(wins, decisive), 5) if decisive else 1.0,
        "significant_at_05": bool(decisive and two_sided_binomial_p(wins, decisive) < 0.05),
        "order_inconsistency_rate": round(sum(1 for o in outcomes if not o.consistent) / n, 4),
        "by_category": by_cat,
        "mean_scores": mean_scores,
        "mean_chars": {
            ARM_TUNED: round(statistics.mean(len_tuned), 1),
            ARM_BASE: round(statistics.mean(len_base), 1),
        },
        "length_ratio_tuned_over_base": round(
            statistics.mean(len_tuned) / statistics.mean(len_base), 3
        )
        if statistics.mean(len_base)
        else 0.0,
        "corr_win_vs_length_delta": round(pearson(length_delta, won), 4),
        "shown_first_distribution": dict(Counter(o.shown_first for o in outcomes)),
    }
