"""SentinelDesk CLI. One command per build phase; every command is idempotent."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.table import Table

from .config import Paths, get_settings
from .logging_utils import console, get_logger, setup_logging

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="SentinelDesk — a DPO-tuned support-resolution agent in a LangGraph pipeline.",
)
log = get_logger(__name__)


@app.callback()
def _root(log_level: str = typer.Option("INFO", "--log-level")) -> None:
    setup_logging(log_level)
    Paths.ensure()


# ------------------------------------------------------------------ phase 0
@app.command("dpo-smoke")
def dpo_smoke(
    base_model: str = typer.Option("", help="defaults to SD_BASE_MODEL"),
    out: Path = typer.Option(Path("reports/phase0_smoke.json"), help="where to write the result"),
    keep: bool = typer.Option(False, help="keep the temporary run directory"),
) -> None:
    """PHASE 0 gate: run the real DPO loop end to end on a toy batch."""
    from .dpo.smoke import run_smoke

    model = base_model or get_settings().base_model
    result = run_smoke(model, keep=keep)
    result["base_model"] = model
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    table = Table(title="Phase 0 — DPO loop smoke test", show_header=False)
    for k, v in result.items():
        table.add_row(k, str(v))
    console.print(table)
    if not result["ok"]:
        raise typer.Exit(1)
    console.print("[green]PASS[/] loss fell and reward margin grew on the toy batch.")


# ------------------------------------------------------------------ phase 1
@app.command("gen-tickets")
def gen_tickets(
    n: int = typer.Option(600, help="how many tickets to generate"),
    heldout_frac: float = typer.Option(0.25, help="fraction reserved for the Phase 5 benchmark"),
    concurrency: int = typer.Option(8),
    out: Path = typer.Option(Path("data/processed/tickets.jsonl")),
) -> None:
    """PHASE 1a: generate synthetic tickets grounded in the policy KB, then split."""
    from .data.generate import build_generator, coverage_report, split_tickets
    from .data.schema import dump_json, write_jsonl

    gen = build_generator()
    tickets = gen.generate(n, concurrency=concurrency)
    tickets = split_tickets(tickets, heldout_frac, get_settings().seed)
    write_jsonl(out, tickets)

    report = coverage_report(tickets)
    dump_json(Paths.reports / "phase1_tickets.json", report)
    table = Table(title=f"Phase 1a — {len(tickets)} tickets -> {out}")
    for k, v in report.items():
        table.add_row(k, json.dumps(v) if isinstance(v, dict) else str(v))
    console.print(table)


@app.command("build-pairs")
def build_pairs_cmd(
    tickets_path: Path = typer.Option(Path("data/processed/tickets.jsonl")),
    split: str = typer.Option("train", help="only tickets in this split become pairs"),
    limit: int = typer.Option(0, help="cap the number of tickets (0 = all)"),
    model: str = typer.Option("", help="generator model path; defaults to SD_BASE_MODEL"),
    batch_size: int = typer.Option(16),
    concurrency: int = typer.Option(8, help="parallel judge calls"),
    min_margin: float = typer.Option(0.0, help="drop pairs whose score gap is below this"),
    out_dir: Path = typer.Option(Path("data/prefs")),
) -> None:
    """PHASE 1b/c: two candidates per ticket, judged in both orders, into DPO pairs."""
    from .data.schema import Ticket, read_jsonl
    from .prefs.build_pairs import build_pairs, persist
    from .prefs.judge import build_judge
    from .serving.local import HFGenerator

    s = get_settings()
    tickets = [t for t in read_jsonl(tickets_path, Ticket) if t.split == split]
    if limit:
        tickets = tickets[:limit]
    if not tickets:
        console.print(f"[red]no tickets with split={split} in {tickets_path}")
        raise typer.Exit(1)

    gen = HFGenerator(model or s.base_model, batch_size=batch_size)
    result = build_pairs(
        tickets, gen, build_judge(), tokenizer=gen.tokenizer,
        concurrency=concurrency, min_margin=min_margin,
    )
    persist(result, out_dir, Paths.reports)

    table = Table(title=f"Phase 1 — {len(result.pairs)} usable pairs from {len(tickets)} tickets")
    for k, v in result.stats.items():
        table.add_row(k, json.dumps(v) if isinstance(v, (dict, list)) else str(v))
    console.print(table)


@app.command("show-ticket")
def show_ticket(
    ticket_id: str = typer.Argument(...),
    path: Path = typer.Option(Path("data/processed/tickets.jsonl")),
) -> None:
    """Print one ticket with the policy context the agent would receive."""
    from .data.kb import render_policies, retrieve_for_ticket
    from .data.schema import Ticket, read_jsonl

    hit = next((t for t in read_jsonl(path, Ticket) if t.id == ticket_id), None)
    if hit is None:
        console.print(f"[red]no ticket {ticket_id} in {path}")
        raise typer.Exit(1)
    console.print(f"[bold]{hit.id}[/] {hit.category}/{hit.urgency} split={hit.split} {hit.source}")
    console.print(f"[bold]Subject:[/] {hit.subject}\n{hit.body}\n")
    console.print("[dim]" + render_policies(retrieve_for_ticket(hit.category, hit.policy_ids)) + "[/dim]")


@app.command("kb")
def kb() -> None:
    """Show the ground-truth policy knowledge base."""
    from .data.kb import load_policies

    table = Table(title="Support policies (ground truth for judging)")
    table.add_column("id")
    table.add_column("category")
    table.add_column("title")
    for p in load_policies():
        table.add_row(p.id, p.category, p.title)
    console.print(table)


@app.command("doctor")
def doctor() -> None:
    """Check that the pieces every phase depends on are reachable."""
    import torch

    s = get_settings()
    table = Table(title="SentinelDesk environment")
    table.add_column("check")
    table.add_column("value")

    table.add_row("torch", torch.__version__)
    table.add_row("device", s.resolved_device())
    table.add_row("base model", s.base_model)
    table.add_row("judge model", s.judge_model)
    table.add_row("judge endpoint", s.judge_base_url)

    from .llm import ChatClient, LLMError

    try:
        with ChatClient(s.judge_base_url, s.judge_api_key, s.judge_model, max_retries=1) as c:
            res = c.chat([{"role": "user", "content": "reply with OK"}], max_tokens=2000)
        table.add_row("judge reachable", f"[green]yes[/] ({res.latency_s:.1f}s)")
    except LLMError as exc:
        table.add_row("judge reachable", f"[red]no[/] ({str(exc)[:60]})")

    console.print(table)


if __name__ == "__main__":
    app()
