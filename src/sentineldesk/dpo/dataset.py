"""Tokenising preference pairs into the (input_ids, labels) tensors DPO needs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from ..data.schema import PreferencePair
from ..prompts import RESOLUTION_SYSTEM
from .loss import IGNORE_INDEX


@dataclass
class TokenizedPair:
    chosen_ids: list[int]
    chosen_labels: list[int]
    rejected_ids: list[int]
    rejected_labels: list[int]


class PreferenceDataset(Dataset):
    """Preference pairs tokenised with the model's own chat template.

    The prompt is rendered through the chat template with a generation prompt so
    that training-time token positions match what the served model sees at
    inference; a mismatch here is invisible in the loss curve and shows up only as
    a tuned model that behaves worse than its own training metrics suggest.
    """

    def __init__(
        self,
        pairs: list[PreferencePair],
        tokenizer,
        *,
        max_prompt_tokens: int = 1024,
        max_response_tokens: int = 512,
        system_prompt: str | None = RESOLUTION_SYSTEM,
    ) -> None:
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.max_prompt_tokens = max_prompt_tokens
        self.max_response_tokens = max_response_tokens
        self.pairs = pairs
        self.examples = [self._encode(p) for p in pairs]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> TokenizedPair:
        return self.examples[idx]

    # ------------------------------------------------------------------ encode
    def _encode_prompt(self, prompt: str) -> list[int]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        # transformers >=5 returns a BatchEncoding here; <5 returns a bare list.
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        if len(ids) > self.max_prompt_tokens:
            # Truncate from the left: the tail of a prompt holds the actual ticket and
            # the generation marker, and dropping the marker breaks the template.
            ids = ids[-self.max_prompt_tokens :]
        return list(ids)

    def _encode_response(self, text: str) -> list[int]:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        ids = list(ids[: self.max_response_tokens - 1])
        eos = self.tokenizer.eos_token_id
        if eos is not None:
            ids.append(eos)
        return ids

    def _encode(self, pair: PreferencePair) -> TokenizedPair:
        prompt_ids = self._encode_prompt(pair.prompt)
        chosen_ids = self._encode_response(pair.chosen)
        rejected_ids = self._encode_response(pair.rejected)
        return TokenizedPair(
            chosen_ids=prompt_ids + chosen_ids,
            chosen_labels=[IGNORE_INDEX] * len(prompt_ids) + chosen_ids,
            rejected_ids=prompt_ids + rejected_ids,
            rejected_labels=[IGNORE_INDEX] * len(prompt_ids) + rejected_ids,
        )


@dataclass
class DPOCollator:
    """Left-pads a batch and stacks chosen+rejected into one 2B forward pass.

    Two decisions here, both about cost rather than correctness of the objective.

    *Stacking* chosen and rejected into a single 2B forward keeps the device busy;
    two separate forwards of size B leave it idle in between, and the reference model
    has to see the same sequences anyway.

    *Left* padding puts every sequence's response tokens flush against the right edge,
    which is what makes ``logits_to_keep`` usable: the loss only needs logits at
    response positions, and computing them for the whole ~1000-token sequence when
    ~150 positions matter is most of the training cost at this vocabulary size.

    Left padding has one trap. Position ids are inferred as 0..T-1 when not supplied,
    which with left padding counts the pad tokens and shifts every real token's RoPE
    position by a per-row amount. Nothing errors; the model just attends with wrong
    positions and the log-probs are quietly wrong. They are computed from the
    attention mask here instead.
    """

    pad_token_id: int

    def __call__(self, batch: list[TokenizedPair]) -> dict[str, torch.Tensor]:
        ids = [ex.chosen_ids for ex in batch] + [ex.rejected_ids for ex in batch]
        labels = [ex.chosen_labels for ex in batch] + [ex.rejected_labels for ex in batch]
        width = max(len(x) for x in ids)

        input_ids = torch.full((len(ids), width), self.pad_token_id, dtype=torch.long)
        label_ids = torch.full((len(ids), width), IGNORE_INDEX, dtype=torch.long)
        attention = torch.zeros((len(ids), width), dtype=torch.long)

        for i, (seq, lab) in enumerate(zip(ids, labels, strict=True)):
            start = width - len(seq)
            input_ids[i, start:] = torch.tensor(seq, dtype=torch.long)
            label_ids[i, start:] = torch.tensor(lab, dtype=torch.long)
            attention[i, start:] = 1

        # 0..n-1 over the real tokens, pads parked at 0 (they are masked out anyway).
        position_ids = (attention.cumsum(dim=-1) - 1).clamp(min=0)

        # How many trailing positions the loss actually needs. The earliest scored
        # label in the batch is at `first`; predicting it needs the logits at `first-1`,
        # so the window runs from there to the end.
        scored = label_ids != IGNORE_INDEX
        first = int(scored.float().argmax(dim=-1).min().item())
        logits_to_keep = min(width, width - first + 1)

        return {
            "input_ids": input_ids,
            "labels": label_ids,
            "attention_mask": attention,
            "position_ids": position_ids,
            "batch_size": torch.tensor(len(batch)),
            "logits_to_keep": torch.tensor(logits_to_keep),
        }
