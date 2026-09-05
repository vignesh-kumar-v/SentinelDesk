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
    balance_length: bool = typer.Option(
        False, help="trim length-skewed pairs until chosen/rejected length is near parity"
    ),
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
        tickets, gen, build_judge(), out_dir=out_dir, tokenizer=gen.tokenizer,
        concurrency=concurrency, min_margin=min_margin,
        balance_length_ratio=balance_length,
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


# ------------------------------------------------------------------ phase 1 gate
@app.command("spotcheck")
def spotcheck(
    n: int = typer.Option(40, help="comparisons to re-judge"),
    second_model: str = typer.Option("kimi-k3:cloud", help="an independent frontier judge"),
    prefs_dir: Path = typer.Option(Path("data/prefs")),
    tickets_path: Path = typer.Option(Path("data/processed/tickets.jsonl")),
    concurrency: int = typer.Option(4),
) -> None:
    """PHASE 1 gate (automated): cross-check the judge against a different frontier model."""
    from ._io import load_judgements
    from .data.schema import dump_json
    from .prefs.spotcheck import build_second_judge, second_judge_agreement

    judgements, tickets, candidates = load_judgements(prefs_dir, tickets_path)
    res = second_judge_agreement(
        judgements, tickets, candidates, build_second_judge(second_model),
        n=n, seed=get_settings().seed, concurrency=concurrency,
    )
    out = res.as_dict() | {
        "primary_judge": get_settings().judge_model,
        "second_judge": second_model,
    }
    dump_json(Paths.reports / "phase1_spotcheck_second_judge.json", out)

    table = Table(title=f"Phase 1 gate — {get_settings().judge_model} vs {second_model}")
    for k in ("n", "agree", "raw_agreement", "cohens_kappa",
              "decisive_n", "decisive_agree", "decisive_agreement", "confusion"):
        table.add_row(k, json.dumps(out[k]) if isinstance(out[k], dict) else str(out[k]))
    console.print(table)
    console.print(f"[dim]{len(out['disagreements'])} disagreements written to the report[/dim]")


@app.command("spotcheck-human")
def spotcheck_human(
    n: int = typer.Option(20),
    worksheet: Path = typer.Option(Path("reports/phase1_spotcheck_human.json")),
    score: bool = typer.Option(False, "--score", help="score an already-filled worksheet"),
    prefs_dir: Path = typer.Option(Path("data/prefs")),
    tickets_path: Path = typer.Option(Path("data/processed/tickets.jsonl")),
) -> None:
    """PHASE 1 gate (human): write a blind worksheet, or score one you filled in.

    Without --score this writes the worksheet: each row shows the ticket and two
    responses as X and Y with the judge's label withheld and the X/Y order
    randomised. Fill in "your_label" on each row ("X", "Y" or "tie"), then re-run
    with --score.
    """
    from ._io import load_judgements
    from .data.schema import dump_json
    from .prefs.spotcheck import make_human_worksheet, score_human_worksheet

    if score:
        res = score_human_worksheet(worksheet)
        out = res.as_dict() | {"judge": get_settings().judge_model, "worksheet": str(worksheet)}
        dump_json(Paths.reports / "phase1_spotcheck_human_result.json", out)
        table = Table(title="Phase 1 gate — human vs judge")
        for k in ("n", "agree", "raw_agreement", "cohens_kappa", "confusion"):
            table.add_row(k, json.dumps(out[k]) if isinstance(out[k], dict) else str(out[k]))
        console.print(table)
        threshold = 0.85
        verdict = "[green]PASS[/]" if res.raw_agreement >= threshold else "[yellow]REVIEW THE RUBRIC[/]"
        console.print(f"{verdict} blueprint gate is agreement >= {threshold:.0%}")
        return

    judgements, tickets, candidates = load_judgements(prefs_dir, tickets_path)
    rows = make_human_worksheet(judgements, tickets, candidates, worksheet, n=n, seed=get_settings().seed)
    console.print(f"wrote {len(rows)} blind comparisons to [bold]{worksheet}[/]")
    console.print('fill in "your_label" on each row ("X", "Y" or "tie"), then:')
    console.print("  sentineldesk spotcheck-human --score")


# ------------------------------------------------------------------ phase 2
@app.command("train-dpo")
def train_dpo(
    pairs: Path = typer.Option(Path("data/prefs/pairs.jsonl")),
    out_dir: Path = typer.Option(Path("artifacts/dpo")),
    beta: float = typer.Option(0.1),
    lr: float = typer.Option(5e-7),
    epochs: int = typer.Option(2),
    batch_size: int = typer.Option(2),
    grad_accum: int = typer.Option(8),
    label_smoothing: float = typer.Option(0.0, help="cDPO; set to the judge's measured error rate"),
    base_model: str = typer.Option(""),
    dtype: str = typer.Option("float32"),
) -> None:
    """PHASE 2: DPO fine-tune the resolution model on the preference pairs."""
    from .dpo.plots import plot_history
    from .dpo.train import DPOConfig, train

    cfg = DPOConfig(
        base_model=base_model or get_settings().base_model,
        output_dir=str(out_dir), beta=beta, learning_rate=lr, epochs=epochs,
        batch_size=batch_size, grad_accum=grad_accum, label_smoothing=label_smoothing,
        seed=get_settings().seed, dtype=dtype,
    )
    summary = train(cfg, pairs)
    (Paths.reports / "phase2_training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    try:
        png = plot_history(out_dir / "history.json", Paths.reports / "phase2_training.png")
        summary["curves"] = str(png)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not plot curves: %s", exc)

    table = Table(title="Phase 2 — DPO training")
    for k in ("checkpoint", "optimiser_steps", "train_pairs", "eval_pairs", "wall_clock_s"):
        table.add_row(k, str(summary[k]))
    for label, key in (("first step", "first_step"), ("last step", "last_step"),
                       ("first eval", "first_eval"), ("last eval", "last_eval")):
        row = summary.get(key) or {}
        if row:
            table.add_row(
                label,
                f"loss {row['loss']:.4f}  margin {row['reward/margin']:+.3f}  "
                f"acc {row['reward/accuracy']:.2f}  kl {row.get('kl/chosen', 0):+.3f}",
            )
    console.print(table)


@app.command("sweep-dpo")
def sweep_dpo(
    pairs: Path = typer.Option(Path("data/prefs/pairs.jsonl")),
    out_root: Path = typer.Option(Path("artifacts/dpo/sweep")),
    lrs: str = typer.Option("5e-7,2e-6,8e-6", help="comma-separated learning rates"),
    betas: str = typer.Option("0.1", help="comma-separated beta values"),
    epochs: int = typer.Option(3),
    batch_size: int = typer.Option(1),
    grad_accum: int = typer.Option(16),
    label_smoothing: float = typer.Option(0.0),
    drift_limit: float = typer.Option(30.0, help="flag runs whose policy drifts past this"),
    base_model: str = typer.Option(""),
) -> None:
    """PHASE 2: sweep DPO hyperparameters, selecting on a validation split.

    ~400 pairs at an effective batch of 16 is about 25 optimiser steps per epoch. The
    5e-7 quoted in the DPO literature assumes thousands of steps on a far larger
    model, so the rate has to be chosen rather than assumed. Selection never touches
    the held-out tickets the Phase 5 arena uses.
    """
    from .dpo.plots import plot_history
    from .dpo.sweep import run_sweep
    from .dpo.train import DPOConfig

    base = DPOConfig(
        base_model=base_model or get_settings().base_model,
        epochs=epochs, batch_size=batch_size, grad_accum=grad_accum,
        label_smoothing=label_smoothing, seed=get_settings().seed,
    )
    grid = [
        {"learning_rate": float(lr), "beta": float(b)}
        for b in betas.split(",")
        for lr in lrs.split(",")
    ]
    report = run_sweep(pairs, base, grid, out_root, drift_limit=drift_limit)

    table = Table(title=f"Phase 2 — DPO sweep ({len(grid)} configs)")
    for col in ("config", "lr", "beta", "steps", "train loss", "eval loss",
                "eval acc", "eval margin", "drift"):
        table.add_column(col)
    for r in report["runs"]:
        mark = "[green]" if r["name"] == report["selected"] else ""
        end = "[/]" if mark else ""
        drift = f"{r['drift_chosen']:.2f}" + (" [red]!over[/]" if r["over_drift_limit"] else "")
        table.add_row(
            f"{mark}{r['name']}{end}", f"{r['lr']:.0e}", str(r["beta"]), str(r["steps"]),
            f"{r['train_loss_first']:.3f}->{r['train_loss_last']:.3f}",
            f"{r['eval_loss']:.4f}", f"{r['eval_accuracy']:.3f}",
            f"{r['eval_margin']:+.3f}", drift,
        )
    console.print(table)
    console.print(f"[green]selected[/] {report['selected']} -> {report['published_checkpoint']}")
    console.print(f"[dim]{report['selection_rule']}[/dim]")

    dump_selected = Paths.reports / "phase2_sweep.json"
    dump_selected.write_text(json.dumps(report, indent=2), encoding="utf-8")
    best_hist = Path(report["selected_checkpoint"]).parent / "history.json"
    try:
        plot_history(best_hist, Paths.reports / "phase2_training.png")
        # the winning run's summary is what the results report reads
        summary = json.loads((Path(report["selected_checkpoint"]).parent / "summary.json").read_text())
        (Paths.reports / "phase2_training_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("could not plot curves: %s", exc)


@app.command("plot-curves")
def plot_curves(
    history: Path = typer.Option(Path("artifacts/dpo/history.json")),
    out: Path = typer.Option(Path("reports/phase2_training.png")),
) -> None:
    """Re-plot the DPO training curves from a saved history."""
    from .dpo.plots import plot_history

    console.print(f"wrote {plot_history(history, out)}")


# ------------------------------------------------------------------ phase 3
@app.command("graph-check")
def graph_check(
    n: int = typer.Option(10),
    split: str = typer.Option("heldout"),
    model: str = typer.Option("", help="checkpoint to resolve with; defaults to the base model"),
    remote: bool = typer.Option(False, help="resolve via the OpenAI-compatible endpoint (vLLM)"),
    tickets_path: Path = typer.Option(Path("data/processed/tickets.jsonl")),
    llm_triage: bool = typer.Option(True),
) -> None:
    """PHASE 3 gate: run sample tickets through the whole graph and check the routing."""
    from .agents.graph import build_pipeline
    from .agents.resolution import LocalResolver, RemoteResolver
    from .data.schema import Ticket, dump_json, read_jsonl
    from .llm import ChatClient

    s = get_settings()
    tickets = [t for t in read_jsonl(tickets_path, Ticket) if t.split == split][:n]
    if remote:
        resolver = RemoteResolver(
            ChatClient(s.resolution_base_url, s.resolution_api_key, s.resolution_model),
            model=s.resolution_model,
        )
    else:
        from .serving.local import HFGenerator

        resolver = LocalResolver(HFGenerator(model or s.base_model, batch_size=1))

    pipe = build_pipeline(resolver, use_llm_triage=llm_triage)
    rows, correct = [], 0
    for t in tickets:
        out = pipe.run(t.id, t.subject, t.body, true_category=t.category)
        correct += int(out.get("category") == t.category)
        rows.append(
            {
                "ticket_id": t.id,
                "true_category": t.category,
                "triaged": out.get("category"),
                "triage_ok": out.get("category") == t.category,
                "triage_source": out.get("triage_source"),
                "urgency": out.get("urgency"),
                "confidence": out.get("confidence"),
                "queue": out.get("queue"),
                "escalation_reasons": out.get("escalation_reasons", []),
                "draft_chars": len(out.get("draft", "")),
                "nodes": [tr["node"] for tr in out.get("trace", [])],
                "total_latency_s": next(
                    (tr["latency_s"] for tr in out.get("trace", []) if tr["node"] == "_total"), None
                ),
            }
        )

    escalated = sum(1 for r in rows if r["queue"] != "auto-resolved")
    summary = {
        "n": len(rows),
        "triage_accuracy": round(correct / len(rows), 3) if rows else 0.0,
        "escalated": escalated,
        "escalation_rate": round(escalated / len(rows), 3) if rows else 0.0,
        "resolver": "remote" if remote else "local",
        "rows": rows,
    }
    dump_json(Paths.reports / "phase3_graph_check.json", summary)

    table = Table(title=f"Phase 3 — {len(rows)} tickets through the graph")
    for col in ("ticket", "true", "triaged", "conf", "queue", "why"):
        table.add_column(col)
    for r in rows:
        mark = "[green]" if r["triage_ok"] else "[red]"
        table.add_row(
            r["ticket_id"], r["true_category"], f"{mark}{r['triaged']}[/]",
            f"{r['confidence']:.2f}", r["queue"],
            (r["escalation_reasons"][0][:44] if r["escalation_reasons"] else ""),
        )
    console.print(table)
    console.print(
        f"triage accuracy [bold]{summary['triage_accuracy']:.0%}[/]  "
        f"escalation rate [bold]{summary['escalation_rate']:.0%}[/]"
    )


# ------------------------------------------------------------------ phase 4
@app.command("serve")
def serve(
    model: Path = typer.Option(Path("artifacts/dpo/checkpoint")),
    port: int = typer.Option(8000),
    name: str = typer.Option("sentineldesk-dpo"),
    dtype: str = typer.Option("bfloat16"),
    max_model_len: int = typer.Option(2048),
    backend: str = typer.Option("docker", help="docker | native (native hangs on macOS)"),
    foreground: bool = typer.Option(True, help="block until interrupted"),
) -> None:
    """PHASE 4: serve a checkpoint through vLLM's OpenAI-compatible server."""
    import time

    from .serving.vllm_server import VLLMServer

    server = VLLMServer(
        model_path=str(model), served_name=name, port=port, dtype=dtype,
        max_model_len=max_model_len, backend=backend,
    )
    server.start()
    console.print(f"[green]serving[/] {model} at {server.base_url} as {name!r}")
    if not foreground:
        return
    console.print("[dim]ctrl-c to stop[/dim]")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()


@app.command("bench")
def bench(
    n: int = typer.Option(32, help="requests per backend"),
    max_tokens: int = typer.Option(160),
    concurrency: int = typer.Option(8, help="for the concurrent vLLM regime"),
    tickets_path: Path = typer.Option(Path("data/processed/tickets.jsonl")),
    vllm_url: str = typer.Option("", help="defaults to SD_RESOLUTION_BASE_URL"),
    vllm_model: str = typer.Option("", help="defaults to SD_RESOLUTION_MODEL"),
    hf_model: str = typer.Option("", help="local checkpoint for the transformers/MLX arms"),
    skip_mlx: bool = typer.Option(False),
    skip_hf: bool = typer.Option(False),
) -> None:
    """PHASE 4 verification: tokens/s and latency across serving backends."""
    from .data.schema import Ticket, dump_json, read_jsonl
    from .llm import ChatClient
    from .prefs.candidates import STRATEGIES, build_messages
    from .serving.bench import bench_hf, bench_mlx, bench_openai_backend

    s = get_settings()
    tickets = [t for t in read_jsonl(tickets_path, Ticket) if t.split == "heldout"][:n]
    prompts = [build_messages(t, STRATEGIES["grounded"]) for t in tickets]
    results = []

    url = vllm_url or s.resolution_base_url
    model_name = vllm_model or s.resolution_model
    client = ChatClient(url, s.resolution_api_key, model_name, max_retries=1)
    try:
        for c in (1, concurrency):
            results.append(
                bench_openai_backend(
                    client, model_name, prompts, backend="vllm-cpu",
                    concurrency=c, max_tokens=max_tokens,
                ).as_dict()
            )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[yellow]vLLM arm skipped:[/] {str(exc)[:150]}")

    local_model = hf_model or s.base_model
    if not skip_hf:
        from .serving.local import HFGenerator

        results.append(bench_hf(HFGenerator(local_model, batch_size=8), prompts, max_tokens=max_tokens).as_dict())
    if not skip_mlx:
        try:
            results.append(bench_mlx(local_model, prompts[: min(8, len(prompts))], max_tokens=max_tokens).as_dict())
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]MLX arm skipped:[/] {str(exc)[:150]}")

    dump_json(Paths.reports / "phase4_serving_bench.json",
              {"n_requests": len(prompts), "max_tokens": max_tokens, "results": results})

    table = Table(title="Phase 4 — serving throughput and latency")
    for col in ("backend", "mode", "reqs", "tok/s", "p50 s", "p95 s", "wall s", "err"):
        table.add_column(col)
    for r in results:
        table.add_row(
            r["backend"], r["mode"], str(r["n_requests"]), f"{r['throughput_tok_s']:.1f}",
            f"{r['latency_p50_s']:.2f}", f"{r['latency_p95_s']:.2f}",
            f"{r['wall_clock_s']:.1f}", str(r["errors"]),
        )
    console.print(table)


# ------------------------------------------------------------------ phase 5
@app.command("arena")
def arena(
    tuned: Path = typer.Option(Path("artifacts/dpo/checkpoint"), help="the DPO-tuned arm"),
    base: str = typer.Option("", help="the base-prompted arm; defaults to SD_BASE_MODEL"),
    tickets_path: Path = typer.Option(Path("data/processed/tickets.jsonl")),
    n: int = typer.Option(0, help="cap held-out tickets (0 = all)"),
    batch_size: int = typer.Option(16),
    concurrency: int = typer.Option(6),
    temperature: float = typer.Option(0.0, help="0 makes both arms deterministic"),
    max_tokens: int = typer.Option(220),
    reuse_responses: Path = typer.Option(
        None, help="re-judge saved responses instead of regenerating them"
    ),
) -> None:
    """PHASE 5: blind, both-orders judge-scored win-rate on held-out tickets.

    Both arms get the same system prompt, the same retrieved policies and the same
    decoding settings. The only difference between them is the weights.
    """
    from .data.schema import Ticket, dump_json, read_jsonl
    from .eval.arena import ARM_BASE, ARM_TUNED, run_arena
    from .prefs.candidates import STRATEGIES, build_messages
    from .prefs.judge import build_judge, judge_metadata
    from .serving.local import HFGenerator

    s = get_settings()
    tickets = [t for t in read_jsonl(tickets_path, Ticket) if t.split == "heldout"]
    if n:
        tickets = tickets[:n]
    prompts = [build_messages(t, STRATEGIES["grounded"]) for t in tickets]

    arms: dict[str, dict[str, str]] = {}
    if reuse_responses and reuse_responses.exists():
        # Generation is deterministic at temperature 0, so re-judging saved responses
        # gives the same comparison without paying for 20 minutes of decoding again.
        saved = json.loads(reuse_responses.read_text(encoding="utf-8"))
        arms = {ARM_TUNED: {k: v[ARM_TUNED] for k, v in saved.items()},
                ARM_BASE: {k: v[ARM_BASE] for k, v in saved.items()}}
        tickets = [t for t in tickets if t.id in saved]
        console.print(f"reusing {len(saved)} saved response pairs from {reuse_responses}")
    else:
        for arm, path in ((ARM_TUNED, str(tuned)), (ARM_BASE, base or s.base_model)):
            console.print(f"generating [bold]{arm}[/] responses with {path}")
            gen = HFGenerator(path, batch_size=batch_size)
            texts = gen.generate(prompts, temperature=temperature, top_p=1.0,
                                 max_tokens=max_tokens)
            arms[arm] = {t.id: text.strip() for t, text in zip(tickets, texts, strict=True)}
            del gen

    result = run_arena(
        tickets, arms[ARM_TUNED], arms[ARM_BASE], build_judge(),
        seed=s.seed, concurrency=concurrency,
    )

    # Refuse to overwrite a larger completed run with a smaller partial one. The arena
    # writes to fixed paths, so a re-run that loses most of its judge calls to a rate
    # limit will silently replace a good result with a worse one — which is exactly
    # what happened once here. Judge failures are skipped rather than raised, so the
    # command otherwise exits 0 and looks successful.
    prior = Paths.reports / "phase5_arena.json"
    if prior.exists():
        try:
            prev_n = json.loads(prior.read_text(encoding="utf-8")).get("n", 0)
        except Exception:  # noqa: BLE001
            prev_n = 0
        if len(result.outcomes) < prev_n:
            backup = Paths.reports / "phase5_arena.partial.json"
            dump_json(backup, result.summary | {"refused_to_overwrite": prev_n})
            console.print(
                f"[red]refusing to overwrite[/] a {prev_n}-ticket result with "
                f"{len(result.outcomes)} tickets — {prev_n - len(result.outcomes)} judge "
                f"calls failed (rate limit?). Partial result written to {backup}."
            )
            raise typer.Exit(1)
    payload = result.summary | judge_metadata() | {
        "tuned_arm": str(tuned), "base_arm": base or s.base_model,
        "temperature": temperature, "max_tokens": max_tokens,
    }
    dump_json(Paths.reports / "phase5_arena.json", payload)
    dump_json(
        Paths.reports / "phase5_arena_outcomes.json",
        [
            {"ticket_id": o.ticket_id, "category": o.category, "winner": o.winner,
             "consistent": o.consistent, "margin": o.margin, "lengths": o.lengths,
             # Per-dimension scores, not just the aggregate. Without these the only
             # available significance test is on the win/loss/tie reduction, which
             # throws away most of the information: a paired test across 149 tickets
             # on the correctness score is far more powerful than a binomial on who won.
             "scores": o.scores}
            for o in result.outcomes
        ],
    )
    dump_json(
        Paths.reports / "phase5_arena_responses.json",
        {t.id: {ARM_TUNED: arms[ARM_TUNED][t.id], ARM_BASE: arms[ARM_BASE][t.id]} for t in tickets},
    )

    table = Table(title="Phase 5 — DPO-tuned vs base-prompted, blind, held-out")
    for k in ("n", "wins_tuned", "losses_tuned", "ties", "win_rate_adjusted",
              "win_rate_adjusted_ci95", "win_rate_decisive", "win_rate_decisive_ci95",
              "binomial_p_vs_50pct", "significant_at_05", "order_inconsistency_rate",
              "mean_chars", "length_ratio_tuned_over_base", "corr_win_vs_length_delta"):
        v = payload.get(k)
        table.add_row(k, json.dumps(v) if isinstance(v, (dict, list)) else str(v))
    console.print(table)

    cat = Table(title="by category")
    for col in ("category", "n", "wins", "ties", "losses", "win rate (adj)"):
        cat.add_column(col)
    for c, row in payload.get("by_category", {}).items():
        cat.add_row(c, str(row["n"]), str(row["wins"]), str(row["ties"]),
                    str(row["losses"]), f"{row['win_rate_adjusted']:.0%}")
    console.print(cat)

    scores = Table(title="mean rubric scores")
    for col in ("arm", "correctness", "completeness", "conciseness", "tone", "total"):
        scores.add_column(col)
    for arm, row in payload.get("mean_scores", {}).items():
        scores.add_row(arm, *[f"{row[d]:.2f}" for d in
                              ("correctness", "completeness", "conciseness", "tone", "total")])
    console.print(scores)


@app.command("arena-aa")
def arena_aa(
    n: int = typer.Option(30, help="held-out tickets to use"),
    model: str = typer.Option("", help="defaults to SD_BASE_MODEL"),
    concurrency: int = typer.Option(4),
    seed_offset: int = typer.Option(1, help="shifts the second arm's sampling seed"),
) -> None:
    """Null test: run the arena with the SAME model in both arms.

    A benchmark that reports a win-rate has to be shown incapable of manufacturing
    one. Two arms from identical weights differ only by sampling noise, so anything
    far from 50% here is the harness leaking a signal — a position bias the
    both-orders swap failed to cancel, an arm-to-slot correlation, or a judge that
    can tell the arms apart some other way. Run this before trusting the real number.
    """
    from .data.schema import Ticket, dump_json, read_jsonl
    from .eval.arena import run_arena
    from .prefs.candidates import STRATEGIES, build_messages
    from .prefs.judge import build_judge
    from .serving.local import HFGenerator

    s = get_settings()
    path = model or s.base_model
    tickets = [t for t in read_jsonl(Path("data/processed/tickets.jsonl"), Ticket)
               if t.split == "heldout"][:n]
    prompts = [build_messages(t, STRATEGIES["grounded"]) for t in tickets]

    gen = HFGenerator(path, batch_size=8)
    # Nonzero temperature on both arms: at temperature 0 the two arms would be
    # byte-identical and every comparison a trivial tie, which tests nothing.
    torch_seed = s.seed
    import torch

    torch.manual_seed(torch_seed)
    a = gen.generate(prompts, temperature=0.7, top_p=0.9, max_tokens=220)
    torch.manual_seed(torch_seed + seed_offset)
    b = gen.generate(prompts, temperature=0.7, top_p=0.9, max_tokens=220)
    del gen

    result = run_arena(
        tickets, {t.id: x.strip() for t, x in zip(tickets, a, strict=True)},
        {t.id: x.strip() for t, x in zip(tickets, b, strict=True)},
        build_judge(), seed=s.seed, concurrency=concurrency,
    )
    payload = result.summary | {"model": path, "note": "A/A null test — same weights both arms"}
    dump_json(Paths.reports / "phase5_arena_aa.json", payload)

    table = Table(title=f"A/A null test — {path} against itself")
    for k in ("n", "wins_tuned", "losses_tuned", "ties", "win_rate_adjusted",
              "win_rate_adjusted_ci95", "binomial_p_vs_50pct", "significant_at_05",
              "order_inconsistency_rate", "corr_win_vs_length_delta"):
        v = payload.get(k)
        table.add_row(k, json.dumps(v) if isinstance(v, (dict, list)) else str(v))
    console.print(table)
    lo, hi = payload["win_rate_adjusted_ci95"]
    if lo <= 0.5 <= hi:
        console.print("[green]PASS[/] 50% sits inside the interval: the harness is not "
                      "manufacturing a winner.")
    else:
        console.print("[red]FAIL[/] the harness prefers one arm even with identical "
                      "weights — do not trust the real arena until this is explained.")


@app.command("compare-samples")
def compare_samples(
    n: int = typer.Option(6),
    tuned: Path = typer.Option(Path("artifacts/dpo/checkpoint")),
    base: str = typer.Option(""),
    max_tokens: int = typer.Option(220),
    out: Path = typer.Option(Path("reports/phase5_samples.md")),
) -> None:
    """Print base vs tuned responses side by side on held-out tickets.

    A win-rate says which arm the judge preferred; it does not say what changed. These
    are the raw generations, so a reader can see whether the tuned model became more
    accurate, merely shorter, or degenerate — three outcomes a single number cannot
    tell apart.
    """
    from .data.kb import render_policies
    from .data.schema import Ticket, read_jsonl
    from .prefs.candidates import STRATEGIES, build_messages
    from .serving.local import HFGenerator

    s = get_settings()
    tickets = [t for t in read_jsonl(Path("data/processed/tickets.jsonl"), Ticket)
               if t.split == "heldout"][:n]
    prompts = [build_messages(t, STRATEGIES["grounded"]) for t in tickets]

    arms = {}
    for label, path in (("tuned", str(tuned)), ("base", base or s.base_model)):
        gen = HFGenerator(path, batch_size=4)
        arms[label] = gen.generate(prompts, temperature=0.0, top_p=1.0, max_tokens=max_tokens)
        del gen

    lines = ["# Base vs DPO-tuned — held-out samples\n",
             "_Greedy decoding, identical prompts and retrieved policies for both arms._\n"]
    for i, t in enumerate(tickets):
        lines += [
            f"\n## {t.id} — {t.category} / {t.urgency}\n",
            f"**Subject:** {t.subject}\n", f"{t.body}\n",
            f"<details><summary>policy the ticket was written from</summary>\n\n"
            f"```\n{render_policies(t.policy_ids)}\n```\n</details>\n",
            f"**Base** ({len(arms['base'][i])} chars)\n\n> "
            + arms["base"][i].strip().replace("\n", "\n> ") + "\n",
            f"**DPO-tuned** ({len(arms['tuned'][i])} chars)\n\n> "
            + arms["tuned"][i].strip().replace("\n", "\n> ") + "\n",
        ]
        console.print(f"[bold]{t.id}[/] {t.category}  base {len(arms['base'][i])} chars  "
                      f"tuned {len(arms['tuned'][i])} chars")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    console.print(f"wrote {out}")


# ------------------------------------------------------------------ phase 6
@app.command("guardrails-eval")
def guardrails_eval(
    no_llm_rails: bool = typer.Option(False, help="deterministic rails only"),
    out: Path = typer.Option(Path("reports/phase6_guardrails.json")),
) -> None:
    """PHASE 6 gate: score the rails against labelled adversarial and benign cases.

    Reports both error directions. A rail that blocks everything has a perfect
    true-positive rate and would take a support queue offline, so the false-positive
    rate on ordinary tickets carries equal weight.
    """
    from .data.schema import dump_json
    from .guardrails.evaluate import evaluate
    from .guardrails.rails import Guardrails

    guards = Guardrails(use_llm_rails=not no_llm_rails)
    result = evaluate(guards)
    dump_json(out, result)

    table = Table(title="Phase 6 — guardrail evaluation")
    for col in ("rail", "attacks", "caught", "TPR", "benign", "wrongly blocked", "FPR"):
        table.add_column(col)
    for r in result["by_rail"]:
        tpr = f"{r['true_positive_rate']:.0%}"
        fpr = f"{r['false_positive_rate']:.0%}"
        table.add_row(
            r["rail"], str(r["attacks"]), str(r["caught"]),
            f"[green]{tpr}[/]" if r["true_positive_rate"] >= 0.9 else f"[yellow]{tpr}[/]",
            str(r["benign"]), str(r["wrongly_blocked"]),
            f"[green]{fpr}[/]" if r["false_positive_rate"] <= 0.1 else f"[red]{fpr}[/]",
        )
    console.print(table)
    console.print(
        f"overall TPR [bold]{result['overall_true_positive_rate']:.0%}[/]  "
        f"overall FPR [bold]{result['overall_false_positive_rate']:.0%}[/]  "
        f"(LLM rails {'on' if result['rail_model_available'] else 'OFF'})"
    )
    for r in result["by_rail"]:
        for f in r["failures"]:
            console.print(f"  [yellow]{r['rail']}/{f['id']}[/] {f['kind']}: "
                          f"{f.get('text', '')}{f.get('got', '')}")


# ------------------------------------------------------------------ phase 7
@app.command("trace-demo")
def trace_demo(
    n: int = typer.Option(5, help="tickets to run through the traced pipeline"),
    model: str = typer.Option("", help="resolution checkpoint; defaults to the tuned one"),
    guards: bool = typer.Option(True, help="include the guardrail nodes"),
    verify: bool = typer.Option(True, help="read the traces back out of Langfuse"),
    out: Path = typer.Option(Path("reports/phase7_tracing.json")),
) -> None:
    """PHASE 7 gate: run tickets through the traced graph, then read the traces back.

    Verification is the point. Instrumenting a pipeline and asserting the SDK was
    called proves nothing; this fetches the trace back from the server and checks that
    every hop is present with its own timing.
    """
    import os

    from .agents.graph import build_pipeline
    from .agents.resolution import LocalResolver
    from .data.schema import Ticket, dump_json, read_jsonl
    from .observability.tracing import Tracer
    from .serving.local import HFGenerator

    tracer = Tracer.create()
    if not tracer.enabled:
        console.print(f"[red]Langfuse not configured:[/] {tracer.reason}")
        console.print("start it with: make langfuse-up   (then copy the keys into .env)")
        raise typer.Exit(1)

    s = get_settings()
    ckpt = model or ("artifacts/dpo/checkpoint" if Path("artifacts/dpo/checkpoint").exists()
                     else s.base_model)
    tickets = [t for t in read_jsonl(Path("data/processed/tickets.jsonl"), Ticket)
               if t.split == "heldout"][:n]

    guard_layer = None
    if guards:
        from .guardrails.rails import Guardrails

        guard_layer = Guardrails()

    pipe = build_pipeline(
        LocalResolver(HFGenerator(ckpt, batch_size=1)),
        use_llm_triage=True, guards=guard_layer, tracer=tracer,
    )

    rows = []
    for t in tickets:
        state = pipe.run(t.id, t.subject, t.body, true_category=t.category)
        hops = [x["node"] for x in state.get("trace", [])]
        rows.append({
            "ticket_id": t.id, "queue": state.get("queue"),
            "confidence": state.get("confidence"), "hops": hops,
            "total_latency_s": next((x["latency_s"] for x in state["trace"]
                                     if x["node"] == "_total"), None),
        })
        console.print(f"  {t.id}  {state.get('queue'):14s} hops={len(hops)}  "
                      f"{rows[-1]['total_latency_s']}s")
    tracer.flush()

    result = {"tickets": rows, "langfuse_host": os.environ.get("LANGFUSE_HOST", ""),
              "checkpoint": ckpt, "guards": guards}

    if verify:
        import time

        import httpx

        time.sleep(6)  # traces are ingested through a queue, not written synchronously
        host = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
        auth = (os.environ["LANGFUSE_PUBLIC_KEY"], os.environ["LANGFUSE_SECRET_KEY"])
        verified = []
        for row in rows:
            try:
                r = httpx.get(f"{host}/api/public/traces", auth=auth,
                              params={"name": f"ticket:{row['ticket_id']}", "limit": 1},
                              timeout=30)
                data = (r.json().get("data") or [{}])[0]
                obs = data.get("observations") or []
                detail = httpx.get(f"{host}/api/public/traces/{data.get('id')}", auth=auth,
                                   timeout=30).json() if data.get("id") else {}
                names = [o.get("name") for o in (detail.get("observations") or [])]
                verified.append({
                    "ticket_id": row["ticket_id"], "trace_id": data.get("id"),
                    "observations_in_trace": len(obs) or len(names),
                    "hop_names": [n for n in names if n],
                    "total_cost": detail.get("totalCost"),
                    "latency_s": detail.get("latency"),
                })
            except Exception as exc:  # noqa: BLE001
                verified.append({"ticket_id": row["ticket_id"], "error": str(exc)[:160]})
        result["verified_from_server"] = verified

        table = Table(title="Phase 7 — traces read back from Langfuse")
        for col in ("ticket", "trace id", "hops recorded", "latency", "nodes"):
            table.add_column(col)
        for v in verified:
            table.add_row(
                v["ticket_id"], (v.get("trace_id") or "—")[:8],
                str(v.get("observations_in_trace", 0)),
                f"{v.get('latency_s') or 0:.2f}s",
                ",".join(n.replace("ticket:", "") for n in v.get("hop_names", []))[:60],
            )
        console.print(table)

    dump_json(out, result)
    console.print(f"wrote {out}")


@app.command("report")
def report(out: Path = typer.Option(Path("reports/RESULTS.md"))) -> None:
    """Regenerate the results write-up from whichever phase reports exist."""
    from .eval.report import build_report

    path = build_report(Paths.reports, out)
    console.print(f"wrote {path}")


if __name__ == "__main__":
    app()
