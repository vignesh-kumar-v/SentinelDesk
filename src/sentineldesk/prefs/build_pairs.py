"""Phase 1: tickets -> two candidates each -> judged -> (prompt, chosen, rejected).

Both expensive stages checkpoint to disk as they complete and skip work already on
disk when re-run. This is not a nicety: the judging stage is ~900 calls against a
hosted frontier model and runs well over an hour, and an all-or-nothing job of that
length loses everything to a laptop sleeping or a session ending. Resumability also
makes the run safely interruptible, so the pair set can be grown in stages.
"""

from __future__ import annotations

import statistics
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ..data.schema import (
    Candidate,
    JudgeVerdict,
    PreferencePair,
    Ticket,
    dump_json,
    iter_jsonl,
    write_jsonl,
)
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


# --------------------------------------------------------------------- caching
def _load_candidates(path: Path) -> dict[tuple[str, str], Candidate]:
    if not path.exists():
        return {}
    try:
        return {(c.strategy, c.ticket_id): c for c in iter_jsonl(path, Candidate)}
    except ValueError as exc:
        log.warning("ignoring unreadable candidate cache: %s", exc)
        return {}


def _append_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(row.model_dump_json() + "\n")


class _JudgementStore:
    """Append-only judgement log, keyed by ticket id.

    Stored as its own record type rather than as the in-memory PairJudgement so the
    file stays readable by a person auditing a label - the rationales and the raw
    per-order verdicts are the whole point of keeping it.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.rows: dict[str, dict] = {}
        if path.exists():
            import json

            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # A run killed mid-write leaves a torn final line; drop it rather
                    # than failing the resume it exists to enable.
                    log.warning("dropping a torn line in %s", path)
                    continue
                self.rows[row["ticket_id"]] = row

    def has(self, ticket_id: str) -> bool:
        return ticket_id in self.rows

    def add(self, j: PairJudgement) -> None:
        import json

        row = {
            "ticket_id": j.ticket_id,
            "winner": j.winner,
            "consistent": j.consistent,
            "margin": j.margin,
            "scores": j.scores,
            "orders": [v.first_shown for v in j.verdicts],
            "raw_winners": [v.winner for v in j.verdicts],
            "rationales": [v.rationale for v in j.verdicts],
        }
        with self._lock:
            self.rows[j.ticket_id] = row
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
                fh.flush()

    def all(self) -> list[PairJudgement]:
        out = []
        for row in self.rows.values():
            out.append(
                PairJudgement(
                    ticket_id=row["ticket_id"],
                    winner=row["winner"],
                    consistent=row["consistent"],
                    margin=row["margin"],
                    verdicts=[
                        JudgeVerdict(
                            ticket_id=row["ticket_id"],
                            first_shown=first,
                            winner=w,
                            rationale=rat,
                        )
                        for first, w, rat in zip(
                            row.get("orders", []),
                            row.get("raw_winners", []),
                            row.get("rationales", []),
                            strict=False,
                        )
                    ],
                    scores=row["scores"],
                )
            )
        return out


# ----------------------------------------------------------------------- build
def build_pairs(
    tickets: list[Ticket],
    generator,
    judge: Judge,
    *,
    out_dir: Path,
    tokenizer=None,
    concurrency: int = 4,
    min_margin: float = 0.0,
) -> PairBuildResult:
    names = list(STRATEGIES)
    cand_path = out_dir / "candidates.jsonl"
    index = _load_candidates(cand_path)

    for name in names:
        todo = [t for t in tickets if (name, t.id) not in index]
        if not todo:
            log.info("%s: all %d candidates already cached", name, len(tickets))
            continue
        log.info("%s: generating %d candidates (%d cached)", name, len(todo), len(tickets) - len(todo))
        fresh = generate_candidates(generator, todo, name, tokenizer=tokenizer)
        _append_jsonl(cand_path, fresh)
        for c in fresh:
            index[(c.strategy, c.ticket_id)] = c

    store = _JudgementStore(out_dir / "judgements.jsonl")
    pending = [t for t in tickets if not store.has(t.id)]
    log.info(
        "judging %d pairs x 2 orders = %d calls (%d already judged)",
        len(pending), 2 * len(pending), len(tickets) - len(pending),
    )
    if pending:
        items = [(t, index[(names[0], t.id)], index[(names[1], t.id)]) for t in pending]
        # Results are written as each completes rather than collected at the end, so
        # an interrupted run keeps everything it paid for.
        for j in judge.judge_stream(items, concurrency=concurrency):
            if j is not None:
                store.add(j)

    judgements = store.all()
    by_id = {t.id: t for t in tickets}
    judgements = [j for j in judgements if j.ticket_id in by_id]

    meta = judge_metadata()
    pairs: list[PreferencePair] = []
    for j in judgements:
        if not j.consistent or j.margin < min_margin:
            continue
        loser = j.loser
        if loser is None:
            continue
        pairs.append(
            PreferencePair(
                ticket_id=j.ticket_id,
                prompt=canonical_prompt(by_id[j.ticket_id]),
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

    by_strategy = {n: [index[(n, t.id)] for t in tickets if (n, t.id) in index] for n in names}
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
            dim_means[name] = {d: round(statistics.mean(v[d] for v in vals), 3) for d in DIMENSIONS}
            dim_means[name]["total"] = round(statistics.mean(total_score(v) for v in vals), 3)

    chosen_len = [len(p.chosen) for p in pairs]
    rejected_len = [len(p.rejected) for p in pairs]
    return {
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
            k: round(statistics.mean(c.n_chars for c in v), 1) for k, v in by_strategy.items() if v
        },
    }


def persist(result: PairBuildResult, out_dir: Path, reports_dir: Path) -> None:
    # candidates.jsonl and judgements.jsonl are append-only caches written during the
    # run; only the derived pair set and the stats are rewritten here.
    write_jsonl(out_dir / "pairs.jsonl", result.pairs)
    dump_json(reports_dir / "phase1_pairs.json", result.stats)
