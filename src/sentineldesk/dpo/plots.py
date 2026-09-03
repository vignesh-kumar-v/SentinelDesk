"""Training curves for the DPO run.

Three panels rather than a single loss plot, because the loss curve alone cannot
distinguish the run you want from the run that has gone wrong. A collapsing policy
and a learning one both show falling loss and a growing margin; the split of the two
implicit rewards and the KL-to-reference are what tell them apart.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_history(history_path: Path, out_path: Path) -> Path:
    hist = json.loads(history_path.read_text(encoding="utf-8"))
    steps = hist["steps"]
    evals = hist.get("evals", [])
    if not steps:
        raise ValueError(f"no steps in {history_path}")

    x = [s["step"] for s in steps]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))

    ax = axes[0]
    ax.plot(x, [s["loss"] for s in steps], label="train", lw=1.4)
    if evals:
        ax.plot([e["step"] for e in evals], [e["loss"] for e in evals], "o-", label="eval", lw=1.4)
    ax.axhline(0.6931, ls=":", c="grey", lw=1)
    ax.annotate("ln 2 (policy = reference)", (x[0], 0.6931), fontsize=7, c="grey",
                xytext=(2, 4), textcoords="offset points")
    ax.set_title("DPO loss")
    ax.set_xlabel("optimiser step")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(x, [s["reward/chosen"] for s in steps], label="chosen", lw=1.4)
    ax.plot(x, [s["reward/rejected"] for s in steps], label="rejected", lw=1.4)
    ax.axhline(0, ls=":", c="grey", lw=1)
    ax.set_title("implicit rewards  β·log(π/π_ref)")
    ax.set_xlabel("optimiser step")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(x, [s["reward/margin"] for s in steps], lw=1.4, c="tab:green", label="train")
    if evals:
        ax.plot([e["step"] for e in evals], [e["reward/margin"] for e in evals], "o-",
                lw=1.4, c="tab:olive", label="eval")
    ax.axhline(0, ls=":", c="grey", lw=1)
    ax.set_title("reward margin  (chosen − rejected)")
    ax.set_xlabel("optimiser step")
    ax.legend(fontsize=8)

    ax = axes[3]
    ax.plot(x, [s["kl/chosen"] for s in steps], label="on chosen", lw=1.4)
    ax.plot(x, [s["kl/rejected"] for s in steps], label="on rejected", lw=1.4)
    ax.set_title("log π − log π_ref  (drift from reference)")
    ax.set_xlabel("optimiser step")
    ax.legend(fontsize=8)

    for a in axes:
        a.grid(alpha=0.25, lw=0.5)
    fig.suptitle("SentinelDesk — DPO training", y=1.02, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path
