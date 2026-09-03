"""Direct Preference Optimization, written out rather than imported.

Reference: Rafailov et al., "Direct Preference Optimization: Your Language Model
is Secretly a Reward Model" (2023).

The objective, for a preference (x, y_w, y_l):

    r_theta(x, y) = beta * ( log pi_theta(y|x) - log pi_ref(y|x) )
    L_DPO         = -log sigmoid( r_theta(x, y_w) - r_theta(x, y_l) )

r_theta is the *implicit* reward DPO never trains explicitly. Its two halves are
worth logging separately: the margin (r_w - r_l) is what the loss optimises, but a
run where both rewards march downward while the margin grows is a policy collapsing
away from the reference, which shows up in the margin curve as success.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def sequence_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    average: bool = False,
) -> torch.Tensor:
    """Sum log p(token) over the response tokens of each sequence.

    logits: (B, T, V) as returned for the full sequence.
    labels: (B, T) with IGNORE_INDEX on prompt and padding positions.

    Returns (B,). The usual next-token shift is applied here, so callers pass the
    unshifted logits and labels straight from the model and the batch.
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError(f"logits {tuple(logits.shape)} and labels {tuple(labels.shape)} disagree")

    logits = logits[:, :-1, :]
    labels = labels[:, 1:]

    mask = labels != IGNORE_INDEX
    # gather() cannot take IGNORE_INDEX, so park masked positions on a valid id and
    # zero their contribution afterwards.
    safe_labels = labels.masked_fill(~mask, 0)

    logprobs = torch.log_softmax(logits.float(), dim=-1)
    token_logprobs = torch.gather(logprobs, dim=2, index=safe_labels.unsqueeze(2)).squeeze(2)
    token_logprobs = token_logprobs * mask

    summed = token_logprobs.sum(dim=-1)
    if average:
        return summed / mask.sum(dim=-1).clamp(min=1)
    return summed


@dataclass
class DPOStats:
    loss: torch.Tensor
    chosen_reward: torch.Tensor
    rejected_reward: torch.Tensor
    margin: torch.Tensor
    accuracy: torch.Tensor
    chosen_logps: torch.Tensor
    rejected_logps: torch.Tensor
    ref_chosen_logps: torch.Tensor
    ref_rejected_logps: torch.Tensor

    def scalars(self) -> dict[str, float]:
        return {
            "loss": self.loss.item(),
            "reward/chosen": self.chosen_reward.mean().item(),
            "reward/rejected": self.rejected_reward.mean().item(),
            "reward/margin": self.margin.mean().item(),
            "reward/accuracy": self.accuracy.item(),
            "logps/chosen": self.chosen_logps.mean().item(),
            "logps/rejected": self.rejected_logps.mean().item(),
        }


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    *,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> DPOStats:
    """The DPO objective plus the diagnostics worth watching during a run.

    label_smoothing > 0 gives the conservative (cDPO) variant: it assumes the
    preference label is flipped with that probability, which caps how hard the loss
    can push on any single pair. Useful when the labeller is an LLM judge with a
    known, measured disagreement rate rather than a ground-truth human.
    """
    chosen_reward = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_reward = beta * (policy_rejected_logps - ref_rejected_logps)
    logits = chosen_reward - rejected_reward

    if label_smoothing > 0:
        losses = (
            -F.logsigmoid(logits) * (1 - label_smoothing)
            - F.logsigmoid(-logits) * label_smoothing
        )
    else:
        losses = -F.logsigmoid(logits)

    return DPOStats(
        loss=losses.mean(),
        chosen_reward=chosen_reward.detach(),
        rejected_reward=rejected_reward.detach(),
        margin=logits.detach(),
        accuracy=(logits.detach() > 0).float().mean(),
        chosen_logps=policy_chosen_logps.detach(),
        rejected_logps=policy_rejected_logps.detach(),
        ref_chosen_logps=ref_chosen_logps.detach(),
        ref_rejected_logps=ref_rejected_logps.detach(),
    )


def sequence_kl(policy_logps: torch.Tensor, ref_logps: torch.Tensor) -> torch.Tensor:
    """Single-sample estimate of KL(pi_theta || pi_ref) on the sampled responses.

    This is E_y~pi_ref[log pi_theta - log pi_ref] evaluated on the preference
    responses rather than fresh samples, so it is a proxy, not the true KL. It is
    still the cheapest early warning for the failure mode that matters: a policy
    drifting so far from the reference that its generations degenerate while the
    margin curve still looks healthy.
    """
    return (policy_logps - ref_logps).mean()
