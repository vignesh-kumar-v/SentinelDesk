"""Typed records that move between phases.

Everything on disk is JSONL of these models. Reading a file therefore validates
it, which is how a malformed generation run gets caught at the phase boundary
instead of halfway through DPO training.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import BaseModel, Field

Category = Literal["billing", "technical", "account_access", "shipping", "product_info"]
Urgency = Literal["low", "medium", "high"]

CATEGORIES: tuple[Category, ...] = (
    "billing",
    "technical",
    "account_access",
    "shipping",
    "product_info",
)


class Policy(BaseModel):
    """One ground-truth support policy. The judge grades correctness against these."""

    id: str
    category: Category
    title: str
    body: str


class Ticket(BaseModel):
    id: str
    category: Category
    urgency: Urgency
    subject: str
    body: str
    policy_ids: list[str] = Field(default_factory=list)
    split: Literal["train", "heldout"] = "train"
    # free-text notes about how the ticket was produced, kept for provenance
    source: str = "synthetic"


class Candidate(BaseModel):
    """One resolution-agent response to a ticket, plus how it was produced."""

    ticket_id: str
    strategy: str
    text: str
    n_tokens: int = 0
    n_chars: int = 0
    latency_s: float = 0.0


class JudgeVerdict(BaseModel):
    """A single judge call's output, before position-bias aggregation."""

    ticket_id: str
    # which candidate was shown first, so the aggregate can undo position bias
    first_shown: str
    winner: Literal["A", "B", "tie"]
    scores: dict[str, dict[str, float]] = Field(default_factory=dict)
    rationale: str = ""
    raw: str = ""


class PreferencePair(BaseModel):
    """The (prompt, chosen, rejected) triple DPO consumes."""

    ticket_id: str
    prompt: str
    chosen: str
    rejected: str
    chosen_strategy: str
    rejected_strategy: str
    margin: float = 0.0
    agreement: Literal["both_orders", "one_order"] = "both_orders"
    rubric_version: str = ""
    judge_model: str = ""


T = TypeVar("T", bound=BaseModel)


def write_jsonl(path: Path, rows: Iterable[BaseModel]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(row.model_dump_json() + "\n")
            n += 1
    return n


def read_jsonl(path: Path, model: type[T]) -> list[T]:
    return list(iter_jsonl(path, model))


def iter_jsonl(path: Path, model: type[T]) -> Iterator[T]:
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield model.model_validate_json(line)
            except Exception as exc:  # noqa: BLE001 - want the line number in the message
                raise ValueError(f"{path}:{lineno} is not a valid {model.__name__}: {exc}") from exc


def dump_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")
