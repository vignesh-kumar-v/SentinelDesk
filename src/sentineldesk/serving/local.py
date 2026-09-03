"""Batched local generation with plain transformers.

This is the pre-vLLM baseline: the Phase 4 comparison needs a number to beat, and
Phase 1 needs to produce ~1000 candidate responses before any server exists.
"""

from __future__ import annotations

import time

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..logging_utils import get_logger

log = get_logger(__name__)


class HFGenerator:
    def __init__(
        self,
        model_path: str,
        *,
        device: str | None = None,
        dtype: torch.dtype = torch.float32,
        batch_size: int = 16,
    ) -> None:
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Decoder-only batched generation requires left padding; with right padding the
        # model continues from pad tokens and the shorter prompts in a batch generate
        # garbage - silently, since nothing errors.
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).to(device)
        self.model.eval()
        self.model.config.use_cache = True
        self.last_stats: dict[str, float] = {}
        log.info("HFGenerator ready: %s on %s (%s)", model_path, device, dtype)

    @torch.no_grad()
    def generate(
        self,
        batches: list[list[dict[str, str]]],
        *,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 256,
        show_progress: bool = True,
    ) -> list[str]:
        outputs: list[str] = []
        total_new = 0
        t0 = time.perf_counter()
        chunks = range(0, len(batches), self.batch_size)
        bar = tqdm(chunks, desc="generate", disable=not show_progress, total=len(list(chunks)))
        for start in bar:
            chunk = batches[start : start + self.batch_size]
            texts = [
                self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                for m in chunk
            ]
            enc = self.tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            gen = self.model.generate(
                **enc,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                max_new_tokens=max_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            new = gen[:, enc["input_ids"].shape[1] :]
            total_new += int((new != self.tokenizer.pad_token_id).sum())
            outputs.extend(self.tokenizer.batch_decode(new, skip_special_tokens=True))
        elapsed = time.perf_counter() - t0
        self.last_stats = {
            "wall_clock_s": elapsed,
            "output_tokens": total_new,
            "tokens_per_s": total_new / elapsed if elapsed else 0.0,
            "requests": len(batches),
            "batch_size": self.batch_size,
        }
        return outputs
