"""Raw-PyTorch DPO training loop.

No TRL, no Trainer: the optimiser step, the reference-policy handling and the
metric bookkeeping are all explicit, because the reference-policy behaviour is the
part of DPO worth being able to inspect.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..data.schema import PreferencePair, read_jsonl
from ..logging_utils import get_logger
from .dataset import DPOCollator, PreferenceDataset
from .loss import dpo_loss, sequence_kl, sequence_logprobs

log = get_logger(__name__)


@dataclass
class DPOConfig:
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    output_dir: str = "artifacts/dpo"
    beta: float = 0.1
    label_smoothing: float = 0.0
    learning_rate: float = 5e-7
    epochs: int = 2
    batch_size: int = 2
    grad_accum: int = 8
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0
    max_prompt_tokens: int = 1024
    max_response_tokens: int = 448
    eval_fraction: float = 0.1
    eval_every: int = 20
    log_every: int = 1
    seed: int = 1337
    device: str = "auto"
    dtype: str = "bfloat16"
    # Precomputing reference log-probs once and freeing the reference model halves
    # peak memory and removes a second forward pass from every step. Set False to
    # keep the reference model resident (needed if you ever make it non-static).
    cache_reference: bool = True

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def torch_dtype(self) -> torch.dtype:
        return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[
            self.dtype
        ]


@dataclass
class TrainingHistory:
    steps: list[dict] = field(default_factory=list)
    evals: list[dict] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"steps": self.steps, "evals": self.evals}, indent=2), encoding="utf-8"
        )


def _lr_at(step: int, total: int, warmup: int, peak: float) -> float:
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return peak * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def _split_logps(logps: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Undo the collator's chosen-then-rejected stacking."""
    return logps[:n], logps[n:]


@torch.no_grad()
def _reference_logps(
    model, loader: DataLoader, device: str, desc: str = "reference pass"
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    model.eval()
    out = []
    for batch in tqdm(loader, desc=desc, leave=False):
        n = int(batch["batch_size"])
        logps = _forward_logps(model, batch, device)
        c, r = _split_logps(logps, n)
        out.append((c.detach().cpu(), r.detach().cpu()))
    return out


def _forward_logps(model, batch: dict, device: str) -> torch.Tensor:
    """One forward pass, scoring only the positions the loss needs.

    ``logits_to_keep`` restricts the LM head to the trailing window the collator
    computed. On a 151,936-token vocabulary the head and the log-prob reduction are
    most of the cost per step, and roughly six sevenths of a padded sequence here is
    prompt whose logits are masked out of the loss anyway.
    """
    keep = int(batch["logits_to_keep"])
    logits = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        position_ids=batch["position_ids"].to(device),
        logits_to_keep=keep,
    ).logits
    return sequence_logprobs(logits, batch["labels"][:, -keep:].to(device))


def train(cfg: DPOConfig, pairs_path: Path) -> dict:
    torch.manual_seed(cfg.seed)
    device = cfg.resolve_device()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("device=%s dtype=%s base=%s", device, cfg.dtype, cfg.base_model)

    pairs = read_jsonl(pairs_path, PreferencePair)
    if not pairs:
        raise ValueError(f"no preference pairs in {pairs_path}")
    n_eval = max(1, int(len(pairs) * cfg.eval_fraction)) if cfg.eval_fraction > 0 else 0
    rng = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(len(pairs), generator=rng).tolist()
    eval_pairs = [pairs[i] for i in perm[:n_eval]]
    train_pairs = [pairs[i] for i in perm[n_eval:]]
    log.info("pairs: %d train / %d eval", len(train_pairs), len(eval_pairs))

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    ds_kwargs = {
        "max_prompt_tokens": cfg.max_prompt_tokens,
        "max_response_tokens": cfg.max_response_tokens,
    }
    train_ds = PreferenceDataset(train_pairs, tok, **ds_kwargs)
    collate = DPOCollator(pad_token_id=tok.pad_token_id)
    # shuffle=False so the cached reference log-probs line up with the batches that
    # consume them; the dataset was already shuffled above.
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate
    )
    eval_loader = None
    if eval_pairs:
        eval_ds = PreferenceDataset(eval_pairs, tok, **ds_kwargs)
        eval_loader = DataLoader(
            eval_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate
        )

    dtype = cfg.torch_dtype()
    policy = AutoModelForCausalLM.from_pretrained(cfg.base_model, dtype=dtype).to(device)
    policy.config.use_cache = False
    ref = AutoModelForCausalLM.from_pretrained(cfg.base_model, dtype=dtype).to(device)
    ref.config.use_cache = False
    for p in ref.parameters():
        p.requires_grad_(False)

    t_ref = time.perf_counter()
    train_ref = _reference_logps(ref, train_loader, device, "reference pass (train)")
    eval_ref = _reference_logps(ref, eval_loader, device, "reference pass (eval)") if eval_loader else []
    log.info("cached reference log-probs in %.1fs", time.perf_counter() - t_ref)
    if cfg.cache_reference:
        del ref
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

    optim = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum)
    total_steps = steps_per_epoch * cfg.epochs
    warmup = max(1, int(total_steps * cfg.warmup_ratio))
    log.info("optimiser steps: %d (warmup %d)", total_steps, warmup)

    history = TrainingHistory()
    hist_path = out_dir / "history.json"
    global_step = 0
    t0 = time.perf_counter()

    for epoch in range(cfg.epochs):
        policy.train()
        optim.zero_grad(set_to_none=True)
        accum: list[dict] = []
        bar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{cfg.epochs}")
        for i, batch in enumerate(bar):
            n = int(batch["batch_size"])
            logps = _forward_logps(policy, batch, device)
            pol_c, pol_r = _split_logps(logps, n)
            ref_c, ref_r = (t.to(device) for t in train_ref[i])

            stats = dpo_loss(
                pol_c, pol_r, ref_c, ref_r, beta=cfg.beta, label_smoothing=cfg.label_smoothing
            )
            (stats.loss / cfg.grad_accum).backward()

            row = stats.scalars()
            row["kl/chosen"] = sequence_kl(pol_c.detach(), ref_c).item()
            row["kl/rejected"] = sequence_kl(pol_r.detach(), ref_r).item()
            accum.append(row)

            is_last = i == len(train_loader) - 1
            if (i + 1) % cfg.grad_accum == 0 or is_last:
                lr = _lr_at(global_step, total_steps, warmup, cfg.learning_rate)
                for g in optim.param_groups:
                    g["lr"] = lr
                gnorm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                optim.step()
                optim.zero_grad(set_to_none=True)

                merged = {k: sum(r[k] for r in accum) / len(accum) for k in accum[0]}
                merged |= {
                    "step": global_step,
                    "epoch": epoch + (i + 1) / len(train_loader),
                    "lr": lr,
                    "grad_norm": float(gnorm),
                    "elapsed_s": time.perf_counter() - t0,
                }
                history.steps.append(merged)
                accum.clear()
                bar.set_postfix(
                    loss=f"{merged['loss']:.4f}",
                    margin=f"{merged['reward/margin']:.3f}",
                    acc=f"{merged['reward/accuracy']:.2f}",
                )

                if eval_loader and (
                    global_step % cfg.eval_every == 0 or global_step == total_steps - 1
                ):
                    ev = evaluate(policy, eval_loader, eval_ref, device, cfg)
                    ev["step"] = global_step
                    history.evals.append(ev)
                    log.info(
                        "step %d | eval loss %.4f margin %.3f acc %.2f kl %.3f",
                        global_step,
                        ev["loss"],
                        ev["reward/margin"],
                        ev["reward/accuracy"],
                        ev["kl/chosen"],
                    )
                    policy.train()
                global_step += 1
                history.save(hist_path)

    ckpt = out_dir / "checkpoint"
    policy.save_pretrained(ckpt, safe_serialization=True)
    tok.save_pretrained(ckpt)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    history.save(hist_path)

    summary = {
        "checkpoint": str(ckpt),
        "optimiser_steps": global_step,
        "train_pairs": len(train_pairs),
        "eval_pairs": len(eval_pairs),
        "wall_clock_s": round(time.perf_counter() - t0, 1),
        "first_step": history.steps[0] if history.steps else {},
        "last_step": history.steps[-1] if history.steps else {},
        "first_eval": history.evals[0] if history.evals else {},
        "last_eval": history.evals[-1] if history.evals else {},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("saved checkpoint to %s", ckpt)
    return summary


@torch.no_grad()
def evaluate(policy, loader: DataLoader, ref_cache, device: str, cfg: DPOConfig) -> dict:
    policy.eval()
    rows = []
    for i, batch in enumerate(loader):
        n = int(batch["batch_size"])
        logps = _forward_logps(policy, batch, device)
        pol_c, pol_r = _split_logps(logps, n)
        ref_c, ref_r = (t.to(device) for t in ref_cache[i])
        stats = dpo_loss(
            pol_c, pol_r, ref_c, ref_r, beta=cfg.beta, label_smoothing=cfg.label_smoothing
        )
        row = stats.scalars()
        row["kl/chosen"] = sequence_kl(pol_c, ref_c).item()
        row["kl/rejected"] = sequence_kl(pol_r, ref_r).item()
        rows.append(row)
    return {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
