"""Loading the Phase 1 artifacts back off disk.

Judgements are persisted as plain JSON rather than as PairJudgement models because
the file is meant to be readable by a person auditing a label. Rehydrating them here
keeps that decision from leaking into every command that needs them.
"""

from __future__ import annotations

import json
from pathlib import Path

from .data.schema import Candidate, JudgeVerdict, Ticket, read_jsonl
from .prefs.judge import PairJudgement


def load_judgements(
    prefs_dir: Path, tickets_path: Path
) -> tuple[list[PairJudgement], dict[str, Ticket], dict[tuple[str, str], Candidate]]:
    rows = json.loads((prefs_dir / "judgements.json").read_text(encoding="utf-8"))
    judgements = [
        PairJudgement(
            ticket_id=r["ticket_id"],
            winner=r["winner"],
            consistent=r["consistent"],
            margin=r["margin"],
            verdicts=[
                JudgeVerdict(
                    ticket_id=r["ticket_id"],
                    first_shown=first,
                    winner=w,
                    rationale=rat,
                )
                for first, w, rat in zip(
                    r.get("orders", []), r.get("raw_winners", []), r.get("rationales", []),
                    strict=False,
                )
            ],
            scores=r["scores"],
        )
        for r in rows
    ]
    tickets = {t.id: t for t in read_jsonl(tickets_path, Ticket)}
    candidates = {
        (c.strategy, c.ticket_id): c
        for c in read_jsonl(prefs_dir / "candidates.jsonl", Candidate)
    }
    return judgements, tickets, candidates
