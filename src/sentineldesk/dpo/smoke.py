"""Phase 0 verification gate: the real training loop, on a toy batch.

The point is not the numbers but the plumbing - chat template, prompt masking,
reference caching, optimiser step, checkpoint save - all exercised in under a
minute before any real preference data exists. Toy pairs are constructed so the
"chosen" side is trivially preferable, so a working loop must drive the reward
margin positive within a handful of steps. If it does not, something in the
masking or the reference handling is wrong, and finding that out here costs a
minute instead of a training run.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from ..data.schema import PreferencePair, write_jsonl
from ..logging_utils import get_logger
from .train import DPOConfig, train

log = get_logger(__name__)

_TOY = [
    ("How do I reset my password?", "Use the reset link in your email; it expires in 60 minutes.", "idk maybe try turning it off and on again lol"),
    ("When will my refund arrive?", "Approved refunds land in 5-7 business days on the original card.", "Refunds are instant and can go to any card you like."),
    ("Is there a Linux desktop app?", "No native Linux build; the web app is the supported path.", "Yes, download the Linux installer from the downloads page."),
    ("Can support extend my trial?", "Trials are 14 days and cannot be extended by support.", "Sure, I can extend it as many times as you want."),
    ("My parcel is 2 days late.", "A parcel is only declared lost after 10 business days past the estimate.", "It is definitely lost, I will refund you immediately."),
    ("Can I get SSO on Pro?", "SSO is Enterprise-only and is not available as a Pro add-on.", "Yes, SSO is available on every plan including Free."),
    ("How long is deletion reversible?", "Deletion is soft for 30 days and reversible by signing in.", "Deletion is instant and permanent, sorry."),
    ("What is the Pro API rate limit?", "600 requests/minute per workspace on Pro.", "There is no rate limit on any plan.")
]


def run_smoke(base_model: str, keep: bool = False, steps_dir: str | None = None) -> dict:
    pairs = [
        PreferencePair(
            ticket_id=f"toy-{i}",
            prompt=p,
            chosen=c,
            rejected=r,
            chosen_strategy="toy_correct",
            rejected_strategy="toy_wrong",
        )
        for i, (p, c, r) in enumerate(_TOY)
    ]

    tmp = Path(steps_dir) if steps_dir else Path(tempfile.mkdtemp(prefix="sd-smoke-"))
    tmp.mkdir(parents=True, exist_ok=True)
    pairs_path = tmp / "toy_pairs.jsonl"
    write_jsonl(pairs_path, pairs)

    cfg = DPOConfig(
        base_model=base_model,
        output_dir=str(tmp / "run"),
        # A toy run needs a visibly moving margin, so the LR is far above what the
        # real run uses (5e-7). Do not copy this value into a real config.
        learning_rate=5e-6,
        epochs=3,
        batch_size=2,
        grad_accum=2,
        eval_fraction=0.25,
        eval_every=2,
        max_prompt_tokens=256,
        max_response_tokens=64,
    )
    summary = train(cfg, pairs_path)

    first, last = summary["first_step"], summary["last_step"]
    result = {
        "ok": bool(last["reward/margin"] > first["reward/margin"] and last["loss"] < first["loss"]),
        "loss_first": round(first["loss"], 4),
        "loss_last": round(last["loss"], 4),
        "margin_first": round(first["reward/margin"], 4),
        "margin_last": round(last["reward/margin"], 4),
        "accuracy_last": round(last["reward/accuracy"], 3),
        "kl_chosen_last": round(last["kl/chosen"], 4),
        "steps": summary["optimiser_steps"],
        "wall_clock_s": summary["wall_clock_s"],
        "checkpoint_saved": (Path(summary["checkpoint"]) / "model.safetensors").exists(),
    }
    log.info("smoke result: %s", json.dumps(result, indent=2))
    if not keep and steps_dir is None:
        shutil.rmtree(tmp, ignore_errors=True)
    return result
