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
