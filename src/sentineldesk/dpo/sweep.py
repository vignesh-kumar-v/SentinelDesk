"""A small DPO hyperparameter sweep with honest model selection.

Why this exists: ~400 preference pairs at an effective batch of 16 is roughly 25
optimiser steps per epoch. The learning rates quoted in the DPO literature (5e-7)
assume thousands of steps on a much larger model; at this scale they move the policy
so little that a null result would say nothing about DPO and everything about the
step budget. Picking the rate has to be part of the method, not a guess.

The selection rule is the part that matters. Configurations are ranked on a
validation split carved out of the *training* pairs — never on the held-out tickets
the Phase 5 arena uses. Choosing a checkpoint by its arena win-rate and then
reporting that same win-rate is selecting on the test set, which turns the headline
number into a measure of how many configurations were tried.

A run that reaches a high margin by drifting far from the reference policy is not a
better run, so drift is reported alongside every score and a configuration that blows
past the drift guard is flagged rather than silently crowned.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from ..logging_utils import get_logger
from .train import DPOConfig, train

log = get_logger(__name__)


@dataclass
class SweepEntry:
    name: str
    config: DPOConfig
    summary: dict

    @property
    def final_eval(self) -> dict:
        return self.summary.get("last_eval") or {}

    @property
    def eval_accuracy(self) -> float:
        return float(self.final_eval.get("reward/accuracy", 0.0))

    @property
    def eval_margin(self) -> float:
        return float(self.final_eval.get("reward/margin", 0.0))

    @property
    def eval_loss(self) -> float:
        return float(self.final_eval.get("loss", float("inf")))

    @property
    def drift(self) -> float:
        return abs(float(self.final_eval.get("kl/chosen", 0.0)))


def run_sweep(
    pairs_path: Path,
    base: DPOConfig,
    grid: list[dict],
    out_root: Path,
    *,
    drift_limit: float = 30.0,
) -> dict:
    """Train one checkpoint per grid point and rank them on the validation split."""
    entries: list[SweepEntry] = []
    for i, overrides in enumerate(grid):
        name = "_".join(f"{k}{v}" for k, v in overrides.items()) or f"cfg{i}"
        cfg = replace(base, **overrides, output_dir=str(out_root / name))
        log.info("[sweep %d/%d] %s", i + 1, len(grid), name)
        summary = train(cfg, pairs_path)
        entries.append(SweepEntry(name=name, config=cfg, summary=summary))

    ranked = sorted(
        entries,
        # Accuracy first because it is the quantity the arena is a noisy proxy for;
        # margin breaks ties, since two configs can separate the same fraction of
        # pairs while one separates them far more confidently.
        key=lambda e: (e.drift > drift_limit, -e.eval_accuracy, -e.eval_margin, e.eval_loss),
    )
    best = ranked[0]
    log.info(
        "selected %s (eval acc %.3f, margin %+.3f, drift %.2f)",
        best.name, best.eval_accuracy, best.eval_margin, best.drift,
    )

    n_eval = int(entries[0].summary.get("eval_pairs", 0)) if entries else 0
    # Standard error of a proportion at p=0.5. With ~34 validation pairs this is
    # about 8.6 points, so two configurations whose eval accuracies differ by less
    # than roughly twice that are not actually distinguishable. The selection rule is
    # still applied — something has to be chosen — but the margin between the winner
    # and the runner-up is reported so a reader can see whether the choice was
    # decisive or a coin flip.
    acc_se = (0.25 / n_eval) ** 0.5 if n_eval else 0.0
    runner_up = ranked[1] if len(ranked) > 1 else None

    report = {
        "pairs": str(pairs_path),
        "drift_limit": drift_limit,
        "eval_pairs": n_eval,
        "eval_accuracy_standard_error": round(acc_se, 4),
        "selected_margin_over_runner_up": (
            round(best.eval_accuracy - runner_up.eval_accuracy, 4) if runner_up else None
        ),
        "selection_is_within_noise": (
            bool(runner_up and abs(best.eval_accuracy - runner_up.eval_accuracy) < 2 * acc_se)
        ),
        "selection_rule": (
            "highest validation preference accuracy, ties broken by margin then loss; "
            "configs exceeding the drift limit are ranked last. Selection uses a split "
            "of the training pairs only — never the held-out arena tickets."
        ),
        "selected": best.name,
        "selected_checkpoint": best.summary["checkpoint"],
        "runs": [
            {
                "name": e.name,
                "lr": e.config.learning_rate,
                "beta": e.config.beta,
                "epochs": e.config.epochs,
                "label_smoothing": e.config.label_smoothing,
                "steps": e.summary["optimiser_steps"],
                "wall_clock_s": e.summary["wall_clock_s"],
                "train_loss_first": round(e.summary["first_step"].get("loss", 0.0), 4),
                "train_loss_last": round(e.summary["last_step"].get("loss", 0.0), 4),
                "train_margin_last": round(e.summary["last_step"].get("reward/margin", 0.0), 4),
                "eval_loss": round(e.eval_loss, 4),
                "eval_accuracy": round(e.eval_accuracy, 4),
                "eval_margin": round(e.eval_margin, 4),
                "drift_chosen": round(e.drift, 4),
                "over_drift_limit": e.drift > drift_limit,
                # Overfitting shows up as a large train-eval margin gap long before
                # it shows up in the drift guard. A run that separates the training
                # pairs far more confidently than the held-out ones has memorised
                # them, and its eval accuracy can still look competitive by luck on a
                # 34-pair split.
                "train_margin_minus_eval_margin": round(
                    e.summary["last_step"].get("reward/margin", 0.0) - e.eval_margin, 4
                ),
                "eval_loss_rose": bool(
                    e.summary.get("first_eval", {}).get("loss") is not None
                    and e.eval_loss > e.summary["first_eval"]["loss"]
                ),
            }
            for e in entries
        ],
    }
    (out_root / "sweep.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Publish the winner at a stable path so downstream phases do not hardcode a
    # grid point that a later sweep would invalidate.
    final = out_root.parent / "checkpoint"
    if final.exists():
        shutil.rmtree(final)
    shutil.copytree(best.summary["checkpoint"], final)
    (out_root.parent / "selected.json").write_text(
        json.dumps({"selected": best.name, "config": asdict(best.config)}, indent=2),
        encoding="utf-8",
    )
    report["published_checkpoint"] = str(final)
    log.info("published %s to %s", best.name, final)
    return report
