"""Tokenisation and collation, checked against a stub tokenizer.

A stub keeps these tests fast and offline, and pins the contract the real
tokenizer has to satisfy: prompt tokens masked, response tokens kept, EOS appended.
"""

from __future__ import annotations

import torch

from sentineldesk.data.schema import PreferencePair
from sentineldesk.dpo.dataset import DPOCollator, PreferenceDataset
from sentineldesk.dpo.loss import IGNORE_INDEX


class StubTokenizer:
    """Character-code tokenizer. Ids are ord(c); the template adds sentinels."""

    eos_token_id = 2
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        text = messages[-1]["content"]
        return [1, *[ord(c) for c in text], 3]

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def _pair(prompt="ab", chosen="cd", rejected="efg") -> PreferencePair:
    return PreferencePair(
        ticket_id="t1",
        prompt=prompt,
        chosen=chosen,
        rejected=rejected,
        chosen_strategy="grounded",
        rejected_strategy="rushed",
    )


def test_prompt_tokens_are_masked_and_response_tokens_are_not():
    ds = PreferenceDataset([_pair()], StubTokenizer())
    ex = ds[0]
    # prompt is [1, 'a', 'b', 3] -> 4 tokens, all masked
    assert ex.chosen_labels[:4] == [IGNORE_INDEX] * 4
    assert ex.chosen_labels[4:] == [ord("c"), ord("d"), 2]
    assert ex.chosen_ids[:4] == [1, ord("a"), ord("b"), 3]
    assert len(ex.chosen_ids) == len(ex.chosen_labels)


def test_collator_left_pads_so_responses_end_flush_right():
    """logits_to_keep only works if every response ends at the sequence's last position."""
    ds = PreferenceDataset([_pair(), _pair(chosen="z", rejected="yyyy")], StubTokenizer())
    batch = DPOCollator(pad_token_id=0)([ds[0], ds[1]])
    ids, labels, attn = batch["input_ids"], batch["labels"], batch["attention_mask"]
    for row in range(ids.shape[0]):
        assert attn[row, -1] == 1                      # nothing padded on the right
        assert labels[row, -1] != IGNORE_INDEX         # last position is always scored
        pad = (attn[row] == 0).sum().item()
        assert torch.all(attn[row, :pad] == 0)         # all padding is on the left
        assert torch.all(attn[row, pad:] == 1)


def test_position_ids_ignore_left_padding():
    """Inferred positions would count the pads and silently shift every RoPE position."""
    ds = PreferenceDataset([_pair(), _pair(prompt="abcdefgh")], StubTokenizer())
    batch = DPOCollator(pad_token_id=0)([ds[0], ds[1]])
    pos, attn = batch["position_ids"], batch["attention_mask"]
    for row in range(pos.shape[0]):
        real = pos[row][attn[row] == 1]
        assert real.tolist() == list(range(len(real)))  # real tokens start at 0


def test_logits_to_keep_covers_every_scored_position():
    ds = PreferenceDataset([_pair(), _pair(chosen="zzzzzz")], StubTokenizer())
    batch = DPOCollator(pad_token_id=0)([ds[0], ds[1]])
    keep = int(batch["logits_to_keep"])
    labels = batch["labels"]
    # every scored label must fall inside the retained window
    outside = labels[:, :-keep] if keep < labels.shape[1] else labels[:, :0]
    assert torch.all(outside == IGNORE_INDEX)
    assert keep <= labels.shape[1]


def test_windowed_logprobs_equal_full_sequence_logprobs():
    """The speed optimisation must be numerically invisible."""
    from sentineldesk.dpo.loss import sequence_logprobs

    ds = PreferenceDataset([_pair(), _pair(chosen="zzz", rejected="w")], StubTokenizer())
    batch = DPOCollator(pad_token_id=0)([ds[0], ds[1]])
    keep = int(batch["logits_to_keep"])
    torch.manual_seed(11)
    logits = torch.randn(batch["input_ids"].shape[0], batch["input_ids"].shape[1], 130)

    full = sequence_logprobs(logits, batch["labels"])
    windowed = sequence_logprobs(logits[:, -keep:, :], batch["labels"][:, -keep:])
    assert torch.allclose(full, windowed, atol=1e-5)


def test_chosen_and_rejected_share_the_same_prompt_prefix():
    ds = PreferenceDataset([_pair()], StubTokenizer())
    ex = ds[0]
    assert ex.chosen_ids[:4] == ex.rejected_ids[:4]


def test_response_truncation_leaves_room_for_eos():
    ds = PreferenceDataset([_pair(chosen="abcdefghij")], StubTokenizer(), max_response_tokens=4)
    ex = ds[0]
    assert ex.chosen_ids[4:] == [ord("a"), ord("b"), ord("c"), 2]


def test_long_prompt_truncates_from_the_left_keeping_the_generation_marker():
    ds = PreferenceDataset([_pair(prompt="abcdefgh")], StubTokenizer(), max_prompt_tokens=4)
    ex = ds[0]
    assert ex.chosen_ids[3] == 3  # template's trailing generation marker survives


def test_collator_stacks_chosen_then_rejected_and_pads():
    ds = PreferenceDataset([_pair(), _pair(chosen="z", rejected="yyyy")], StubTokenizer())
    batch = DPOCollator(pad_token_id=0)([ds[0], ds[1]])
    assert int(batch["batch_size"]) == 2
    assert batch["input_ids"].shape[0] == 4  # 2 chosen + 2 rejected
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    # padding is masked out of both attention and loss
    pad_positions = batch["attention_mask"] == 0
    assert torch.all(batch["labels"][pad_positions] == IGNORE_INDEX)
    assert torch.all(batch["input_ids"][pad_positions] == 0)


def test_collator_row_order_matches_split_convention():
    """Rows [0:n] must be chosen and [n:] rejected, or the loss pairs the wrong rows."""
    ds = PreferenceDataset([_pair(chosen="c", rejected="rrrr")], StubTokenizer())
    batch = DPOCollator(pad_token_id=0)([ds[0]])
    n = int(batch["batch_size"])
    chosen_len = int(batch["attention_mask"][:n].sum())
    rejected_len = int(batch["attention_mask"][n:].sum())
    assert rejected_len > chosen_len
