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
    """Pads a batch and stacks chosen+rejected into one 2B forward pass.

    Stacking matters on a single accelerator: two separate forwards of size B leave
    the device idle between them, and the reference model has to be run over the same
    sequences anyway.
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
            input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            label_ids[i, : len(lab)] = torch.tensor(lab, dtype=torch.long)
            attention[i, : len(seq)] = 1

        return {
            "input_ids": input_ids,
            "labels": label_ids,
            "attention_mask": attention,
            "batch_size": torch.tensor(len(batch)),
        }
