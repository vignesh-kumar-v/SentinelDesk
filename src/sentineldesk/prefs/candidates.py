"""Generating the two candidate resolutions that become a preference pair.

Both candidates come from the *base model being tuned*, not from a stronger teacher.
That keeps the preference data on-policy: DPO's implicit reward is only meaningful
over responses the policy could plausibly have produced, and pairs where "chosen"
is a frontier model's output teach distillation dressed up as preference learning.

The two strategies differ only in how the model is prompted and sampled. The prompt
recorded in the preference pair is the canonical one from prompts.py for both sides,
because DPO conditions chosen and rejected on a single shared prompt.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from ..data.kb import retrieve_for_ticket
from ..data.schema import Candidate, Ticket
from ..logging_utils import get_logger
from ..prompts import RESOLUTION_SYSTEM, resolution_user_prompt

log = get_logger(__name__)

# The "rushed" strategy is written to produce the specific failure mode this project
# wants DPO to unlearn: eager agreement with the customer, invented specifics, and
# padding. It is not merely "a worse prompt" - it is a targeted one.
RUSHED_SYSTEM = (
    "You are a friendly customer support bot. Your goal is to make the customer happy "
    "and keep them from churning. Be warm and generous, reassure them, and say yes "
    "wherever you can. Write a full, welcoming reply with an opening line and a warm "
    "sign-off."
)


@dataclass(frozen=True)
class Strategy:
    name: str
    system: str
    temperature: float
    top_p: float
    max_tokens: int


STRATEGIES: dict[str, Strategy] = {
    "grounded": Strategy("grounded", RESOLUTION_SYSTEM, 0.6, 0.9, 220),
    "rushed": Strategy("rushed", RUSHED_SYSTEM, 1.0, 0.95, 260),
}


class Generator(Protocol):
    """Anything that can turn batches of chat messages into text."""

    def generate(
        self,
        batches: list[list[dict[str, str]]],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> list[str]: ...


def build_messages(ticket: Ticket, strategy: Strategy) -> list[dict[str, str]]:
    policy_ids = retrieve_for_ticket(ticket.category, ticket.policy_ids)
    return [
        {"role": "system", "content": strategy.system},
        {"role": "user", "content": resolution_user_prompt(ticket.subject, ticket.body, policy_ids)},
    ]


def canonical_prompt(ticket: Ticket) -> str:
    """The single prompt both candidates are trained against."""
    return resolution_user_prompt(
        ticket.subject, ticket.body, retrieve_for_ticket(ticket.category, ticket.policy_ids)
    )


def generate_candidates(
    generator: Generator,
    tickets: list[Ticket],
    strategy_name: str,
    *,
    tokenizer=None,
) -> list[Candidate]:
    strategy = STRATEGIES[strategy_name]
    batches = [build_messages(t, strategy) for t in tickets]
    t0 = time.perf_counter()
    texts = generator.generate(
        batches,
        temperature=strategy.temperature,
        top_p=strategy.top_p,
        max_tokens=strategy.max_tokens,
    )
    elapsed = time.perf_counter() - t0
    per_item = elapsed / max(len(tickets), 1)

    out = []
    for ticket, text in zip(tickets, texts, strict=True):
        text = text.strip()
        n_tokens = len(tokenizer(text)["input_ids"]) if tokenizer else len(text.split())
        out.append(
            Candidate(
                ticket_id=ticket.id,
                strategy=strategy_name,
                text=text,
                n_tokens=n_tokens,
                n_chars=len(text),
                latency_s=per_item,
            )
        )
    log.info(
        "%s: %d candidates in %.1fs (%.2fs/ticket, median %d chars)",
        strategy_name,
        len(out),
        elapsed,
        per_item,
        sorted(c.n_chars for c in out)[len(out) // 2] if out else 0,
    )
    return out
