"""Properties the DPO objective must hold, checked without touching a real model."""

from __future__ import annotations

import math

import pytest
import torch

from sentineldesk.dpo.loss import IGNORE_INDEX, dpo_loss, sequence_kl, sequence_logprobs


def test_sequence_logprobs_masks_prompt_and_padding():
    torch.manual_seed(0)
    b, t, v = 2, 6, 11
    logits = torch.randn(b, t, v)
    labels = torch.full((b, t), IGNORE_INDEX)
    # response occupies positions 3..5 of row 0 and 4..5 of row 1
    labels[0, 3:] = torch.tensor([1, 2, 3])
    labels[1, 4:] = torch.tensor([4, 5])

    got = sequence_logprobs(logits, labels)

    logprobs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    expect0 = sum(logprobs[0, j, labels[0, j + 1]] for j in range(2, 5))
    expect1 = sum(logprobs[1, j, labels[1, j + 1]] for j in range(3, 5))
    assert got[0] == pytest.approx(expect0.item(), abs=1e-5)
    assert got[1] == pytest.approx(expect1.item(), abs=1e-5)


def test_sequence_logprobs_average_normalises_by_length():
    logits = torch.randn(1, 5, 7)
    labels = torch.full((1, 5), IGNORE_INDEX)
    labels[0, 2:] = torch.tensor([1, 2, 3])
    summed = sequence_logprobs(logits, labels)
    averaged = sequence_logprobs(logits, labels, average=True)
    # 3 label positions, one of which is consumed as the shift target's context
    assert averaged.item() == pytest.approx(summed.item() / 3, abs=1e-5)


def test_sequence_logprobs_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        sequence_logprobs(torch.randn(2, 5, 7), torch.zeros(2, 4, dtype=torch.long))


def test_loss_is_log2_when_policy_equals_reference():
    """At init the policy *is* the reference, so every margin is 0 and loss is ln 2."""
    lp = torch.tensor([-10.0, -12.0])
    lr = torch.tensor([-11.0, -9.0])
    stats = dpo_loss(lp, lr, lp, lr, beta=0.1)
    assert stats.loss.item() == pytest.approx(math.log(2), abs=1e-6)
    assert stats.margin.abs().max().item() == pytest.approx(0.0, abs=1e-6)
    assert stats.accuracy.item() == 0.0  # margin > 0 is strict


def test_loss_decreases_as_margin_grows():
    ref_c, ref_r = torch.tensor([-10.0]), torch.tensor([-10.0])
    losses = []
    for delta in (0.0, 1.0, 5.0, 20.0):
        stats = dpo_loss(ref_c + delta, ref_r, ref_c, ref_r, beta=0.1)
        losses.append(stats.loss.item())
    assert losses == sorted(losses, reverse=True)
    assert losses[-1] < 0.2


def test_rewards_are_beta_scaled_logratios():
    beta = 0.25
    stats = dpo_loss(
        torch.tensor([-4.0]), torch.tensor([-9.0]),
        torch.tensor([-6.0]), torch.tensor([-7.0]), beta=beta,
    )
    assert stats.chosen_reward.item() == pytest.approx(beta * 2.0)
    assert stats.rejected_reward.item() == pytest.approx(beta * -2.0)
    assert stats.margin.item() == pytest.approx(beta * 4.0)
    assert stats.accuracy.item() == 1.0


def test_label_smoothing_floors_the_loss():
    """cDPO must not let a single confident pair drive the loss to zero."""
    far = torch.tensor([50.0])
    zero = torch.tensor([0.0])
    plain = dpo_loss(far, zero, zero, zero, beta=0.1, label_smoothing=0.0)
    smooth = dpo_loss(far, zero, zero, zero, beta=0.1, label_smoothing=0.2)
    assert plain.loss.item() < 1e-2
    assert smooth.loss.item() > plain.loss.item()
    # loss floor is -eps*log(sigmoid(-margin)) which grows with the margin
    assert smooth.loss.item() == pytest.approx(0.2 * 5.0, abs=0.05)


def test_gradient_pushes_chosen_up_and_rejected_down():
    pc = torch.tensor([-8.0], requires_grad=True)
    pr = torch.tensor([-8.0], requires_grad=True)
    ref = torch.tensor([-8.0])
    dpo_loss(pc, pr, ref, ref, beta=0.1).loss.backward()
    # gradient descent moves opposite the gradient, so chosen must have negative grad
    assert pc.grad.item() < 0
    assert pr.grad.item() > 0


def test_sequence_kl_sign():
    assert sequence_kl(torch.tensor([-5.0]), torch.tensor([-6.0])).item() == pytest.approx(1.0)
    assert sequence_kl(torch.tensor([-6.0]), torch.tensor([-6.0])).item() == pytest.approx(0.0)


def test_chunking_does_not_change_the_result():
    """The memory fix must be numerically invisible, or it is not a fix."""
    torch.manual_seed(3)
    logits = torch.randn(3, 40, 97)
    labels = torch.full((3, 40), IGNORE_INDEX)
    labels[:, 12:] = torch.randint(0, 97, (3, 28))

    whole = sequence_logprobs(logits, labels, chunk_size=10_000)
    chunked = sequence_logprobs(logits, labels, chunk_size=7)
    assert torch.allclose(whole, chunked, atol=1e-4)


def test_logprobs_match_an_explicit_log_softmax():
    """Pins the identity the memory-efficient path relies on:
    log p(y) == logit[y] - logsumexp(logits)."""
    torch.manual_seed(4)
    logits = torch.randn(2, 9, 23)
    labels = torch.full((2, 9), IGNORE_INDEX)
    labels[:, 4:] = torch.randint(0, 23, (2, 5))

    got = sequence_logprobs(logits, labels)

    reference = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    mask = labels[:, 1:] != IGNORE_INDEX
    picked = torch.gather(
        reference, 2, labels[:, 1:].masked_fill(~mask, 0).unsqueeze(2)
    ).squeeze(2)
    expect = (picked * mask).sum(dim=-1)
    assert torch.allclose(got, expect, atol=1e-4)


def test_logprobs_never_allocate_a_full_vocab_sized_softmax():
    """Regression guard for the OOM: a realistic vocab must not blow up.

    Qwen2.5's vocabulary is 151,936. A (4, 600, 151936) log_softmax in float32 is
    ~1.4 GB on top of the logits themselves, which is what exhausted 24 GB of unified
    memory at batch size 2. This runs the same shape at a size that would be obvious
    if the full-softmax path came back.
    """
    b, t, v = 2, 600, 151_936
    logits = torch.zeros(b, t, v)  # zeros so the allocation is the only cost
    labels = torch.full((b, t), IGNORE_INDEX)
    labels[:, 300:] = 1
    out = sequence_logprobs(logits, labels, chunk_size=64)
    # uniform logits -> every token has probability 1/v.
    # labels[:, 300:] marks 300 positions; the next-token shift keeps all 300 of them
    # as targets (shifted indices 299..598), so the sum is over 300 tokens.
    assert out.shape == (b,)
    assert out[0].item() == pytest.approx(300 * math.log(1 / v), rel=1e-4)
