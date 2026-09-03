"""Phase 1: tickets -> two candidates each -> judged -> (prompt, chosen, rejected)."""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ..data.schema import Candidate, PreferencePair, Ticket, dump_json, write_jsonl
from ..logging_utils import get_logger
from .candidates import STRATEGIES, canonical_prompt, generate_candidates
from .judge import Judge, PairJudgement, judge_metadata
from .rubric import DIMENSIONS, total_score

log = get_logger(__name__)


@dataclass
class PairBuildResult:
    pairs: list[PreferencePair]
    candidates: list[Candidate]
    judgements: list[PairJudgement]
    stats: dict


def build_pairs(
    tickets: list[Ticket],
    generator,
    judge: Judge,
    *,
    tokenizer=None,
    concurrency: int = 4,
    min_margin: float = 0.0,
) -> PairBuildResult:
    names = list(STRATEGIES)
    by_strategy: dict[str, list[Candidate]] = {}
    for name in names:
        by_strategy[name] = generate_candidates(generator, tickets, name, tokenizer=tokenizer)

    index = {(c.strategy, c.ticket_id): c for cs in by_strategy.values() for c in cs}
    items = [(t, index[(names[0], t.id)], index[(names[1], t.id)]) for t in tickets]

    log.info("judging %d pairs x 2 orders = %d judge calls", len(items), 2 * len(items))
    judgements = [j for j in judge.judge_many(items, concurrency=concurrency) if j is not None]

    meta = judge_metadata()
    pairs: list[PreferencePair] = []
    for j in judgements:
        if not j.consistent or j.margin < min_margin:
            continue
        ticket = next(t for t in tickets if t.id == j.ticket_id)
        loser = j.loser
        assert loser is not None
        pairs.append(
            PreferencePair(
                ticket_id=j.ticket_id,
                prompt=canonical_prompt(ticket),
                chosen=index[(j.winner, j.ticket_id)].text,
                rejected=index[(loser, j.ticket_id)].text,
                chosen_strategy=j.winner,
                rejected_strategy=loser,
                margin=j.margin,
                agreement="both_orders",
                rubric_version=meta["rubric_version"],
                judge_model=meta["judge_model"],
            )
        )

    return PairBuildResult(
        pairs=pairs,
        candidates=[c for cs in by_strategy.values() for c in cs],
        judgements=judgements,
        stats=summarise(tickets, by_strategy, judgements, pairs, meta),
    )


def summarise(
    tickets: list[Ticket],
    by_strategy: dict[str, list[Candidate]],
    judgements: list[PairJudgement],
    pairs: list[PreferencePair],
    meta: dict,
) -> dict:
    n = len(judgements)
    inconsistent = sum(1 for j in judgements if not j.consistent)
    winners = Counter(j.winner for j in judgements)

    dim_means: dict[str, dict[str, float]] = {}
    for name in by_strategy:
        vals = [j.scores[name] for j in judgements if name in j.scores]
        if vals:
            dim_means[name] = {
                d: round(statistics.mean(v[d] for v in vals), 3) for d in DIMENSIONS
            }
            dim_means[name]["total"] = round(statistics.mean(total_score(v) for v in vals), 3)

    def lens(side: str) -> list[int]:
        return [len(getattr(p, side)) for p in pairs]

    chosen_len, rejected_len = lens("chosen"), lens("rejected")
    stats = {
        **meta,
        "tickets": len(tickets),
        "candidates_per_ticket": len(by_strategy),
        "judged": n,
        # The headline reliability number: how often swapping the display order
        # flipped the verdict. This bounds how much any downstream win-rate means.
        "order_inconsistency_rate": round(inconsistent / n, 4) if n else 0.0,
        "winner_distribution": dict(winners),
        "usable_pairs": len(pairs),
        "pair_yield": round(len(pairs) / n, 4) if n else 0.0,
        "mean_scores_by_strategy": dim_means,
        "strategy_as_chosen": dict(Counter(p.chosen_strategy for p in pairs)),
        "mean_margin": round(statistics.mean(p.margin for p in pairs), 3) if pairs else 0.0,
        # Length bias check: if chosen is systematically much longer or shorter than
        # rejected, DPO will learn length before it learns correctness.
        "mean_chars_chosen": round(statistics.mean(chosen_len), 1) if pairs else 0,
        "mean_chars_rejected": round(statistics.mean(rejected_len), 1) if pairs else 0,
        "length_ratio_chosen_over_rejected": round(
            statistics.mean(chosen_len) / statistics.mean(rejected_len), 3
        )
        if pairs and statistics.mean(rejected_len)
        else 0.0,
        "mean_candidate_chars": {
            k: round(statistics.mean(c.n_chars for c in v), 1) for k, v in by_strategy.items()
        },
    }
    return stats


def persist(result: PairBuildResult, out_dir: Path, reports_dir: Path) -> None:
    write_jsonl(out_dir / "pairs.jsonl", result.pairs)
    write_jsonl(out_dir / "candidates.jsonl", result.candidates)
    dump_json(
        out_dir / "judgements.json",
        [
            {
                "ticket_id": j.ticket_id,
                "winner": j.winner,
                "consistent": j.consistent,
                "margin": j.margin,
                "scores": j.scores,
                "rationales": [v.rationale for v in j.verdicts],
                "orders": [v.first_shown for v in j.verdicts],
                "raw_winners": [v.winner for v in j.verdicts],
            }
            for j in result.judgements
        ],
    )
    dump_json(reports_dir / "phase1_pairs.json", result.stats)
