"""LLM-as-judge with position-bias control.

Every comparison is run twice with the two candidates swapped. LLM judges have a
well-documented bias toward whichever response is shown first; a single-order label
folds that bias straight into the training data, where it is indistinguishable from
signal. Running both orders costs 2x the judge calls and buys two things:

  * a label only survives if the judge picks the same candidate in both orders, and
  * the disagreement rate is itself a measured, reportable number - the judge's
    order-sensitivity, which sets a ceiling on how much any downstream win-rate
    can be trusted.
"""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from ..config import get_settings
from ..data.kb import render_policies, retrieve_for_ticket
from ..data.schema import Candidate, JudgeVerdict, Ticket
from ..llm import ChatClient, LLMError
from ..logging_utils import get_logger
from .rubric import (
    DIMENSIONS,
    JUDGE_SYSTEM,
    JUDGE_TEMPLATE,
    MAX_SCORES,
    RUBRIC,
    RUBRIC_VERSION,
    rubric_fingerprint,
    total_score,
)

log = get_logger(__name__)


@dataclass
class PairJudgement:
    """Both orders of one comparison, aggregated."""

    ticket_id: str
    winner: str  # a candidate strategy name, or "tie"
    consistent: bool  # did the two orders agree
    margin: float  # mean |total(winner) - total(loser)| across orders
    verdicts: list[JudgeVerdict]
    scores: dict[str, dict[str, float]]  # strategy -> mean dimension scores

    @property
    def loser(self) -> str | None:
        if self.winner == "tie":
            return None
        return next(s for s in self.scores if s != self.winner)


def _clip(raw: dict, key: str) -> dict[str, float]:
    block = raw.get(key) or {}
    return {d: max(0.0, min(float(block.get(d, 0) or 0), MAX_SCORES[d])) for d in DIMENSIONS}


@dataclass
class Judge:
    client: ChatClient
    model: str
    temperature: float = 0.0
    max_tokens: int = 3000

    def _one_order(
        self, ticket: Ticket, first: Candidate, second: Candidate
    ) -> JudgeVerdict:
        prompt = JUDGE_TEMPLATE.format(
            rubric=RUBRIC,
            policies=render_policies(retrieve_for_ticket(ticket.category, ticket.policy_ids)),
            subject=ticket.subject,
            body=ticket.body,
            response_a=first.text or "(empty response)",
            response_b=second.text or "(empty response)",
        )
        obj, res = self.client.chat_json(
            [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}],
            required_keys=("winner",),
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        winner = str(obj.get("winner", "tie")).strip().upper()
        if winner not in {"A", "B"}:
            winner = "tie"
        return JudgeVerdict(
            ticket_id=ticket.id,
            first_shown=first.strategy,
            winner=winner,  # type: ignore[arg-type]
            scores={"A": _clip(obj, "a"), "B": _clip(obj, "b")},
            rationale=str(obj.get("reason", ""))[:400],
            raw=res.text[:1500],
        )

    def judge_pair(self, ticket: Ticket, x: Candidate, y: Candidate) -> PairJudgement:
        v1 = self._one_order(ticket, x, y)  # x is A
        v2 = self._one_order(ticket, y, x)  # y is A
        return aggregate(ticket.id, x.strategy, y.strategy, v1, v2)

    def _safe_judge(self, item: tuple[Ticket, Candidate, Candidate]) -> PairJudgement | None:
        try:
            return self.judge_pair(*item)
        except LLMError as exc:
            log.warning("judge failed for %s: %s", item[0].id, str(exc)[:120])
            return None

    def judge_many(
        self,
        items: list[tuple[Ticket, Candidate, Candidate]],
        *,
        concurrency: int = 4,
    ) -> list[PairJudgement | None]:
        """Judge everything, preserving input order. Use judge_stream to checkpoint."""
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            return list(pool.map(self._safe_judge, items))

    def judge_stream(
        self,
        items: list[tuple[Ticket, Candidate, Candidate]],
        *,
        concurrency: int = 4,
    ) -> Iterator[PairJudgement | None]:
        """Yield verdicts as they complete, so the caller can persist incrementally.

        Order is completion order, not input order. Callers that need input order
        should use judge_many; callers that need to survive an interrupted run - which
        at ~900 hosted-model calls is every caller that matters - should use this.
        """
        done = 0
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(self._safe_judge, item) for item in items]
            for fut in as_completed(futures):
                done += 1
                if done % 25 == 0 or done == len(items):
                    log.info("judged %d/%d", done, len(items))
                yield fut.result()


def aggregate(
    ticket_id: str, sx: str, sy: str, v1: JudgeVerdict, v2: JudgeVerdict
) -> PairJudgement:
    """Fold the two orders into one verdict, undoing the A/B position labels."""
    # v1 showed sx as A; v2 showed sy as A.
    pick1 = sx if v1.winner == "A" else sy if v1.winner == "B" else "tie"
    pick2 = sy if v2.winner == "A" else sx if v2.winner == "B" else "tie"

    scores: dict[str, dict[str, float]] = {}
    for strategy, slots in ((sx, (("A", v1), ("B", v2))), (sy, (("B", v1), ("A", v2)))):
        scores[strategy] = {
            d: sum(v.scores[slot][d] for slot, v in slots) / len(slots) for d in DIMENSIONS
        }

    consistent = pick1 == pick2 and pick1 != "tie"
    winner = pick1 if consistent else "tie"
    totals = {s: total_score(sc) for s, sc in scores.items()}
    margin = abs(totals[sx] - totals[sy])

    return PairJudgement(
        ticket_id=ticket_id,
        winner=winner,
        consistent=consistent,
        margin=margin,
        verdicts=[v1, v2],
        scores=scores,
    )


def build_judge() -> Judge:
    s = get_settings()
    return Judge(
        client=ChatClient(
            s.judge_base_url, s.judge_api_key, s.judge_model, max_retries=s.judge_max_retries
        ),
        model=s.judge_model,
        temperature=s.judge_temperature,
    )


def judge_metadata() -> dict[str, str]:
    s = get_settings()
    return {
        "judge_model": s.judge_model,
        "rubric_version": RUBRIC_VERSION,
        "rubric_fingerprint": rubric_fingerprint(),
    }
