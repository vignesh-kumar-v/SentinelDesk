"""Canonical prompts.

The resolution agent's system prompt and user-prompt layout live here and nowhere
else. Candidate generation (Phase 1), DPO tokenisation (Phase 2) and the served
agent (Phase 3/4) all build their prompts from this module, so the token positions
the model trains on are the ones it later sees at inference. A drift between those
two is invisible in the loss curve and only shows up as a tuned model that
underperforms its own training metrics.
"""

from __future__ import annotations

from .data.kb import render_policies

RESOLUTION_SYSTEM = (
    "You are a support agent for Nimbus. Answer the customer's ticket using only the "
    "policy excerpts provided. Be direct and concise: lead with the answer, then the "
    "concrete next step. Do not invent policy, do not promise anything the excerpts do "
    "not support, and if the excerpts do not cover the question, say so and offer to "
    "escalate. Address the customer directly and keep it under 120 words."
)


def resolution_user_prompt(subject: str, body: str, policy_ids: list[str]) -> str:
    """The user turn the resolution agent sees. Shared by training and serving."""
    return (
        f"POLICY EXCERPTS\n{render_policies(policy_ids)}\n\n"
        f"CUSTOMER TICKET\nSubject: {subject}\n{body.strip()}\n\n"
        "Write the reply to send to this customer."
    )


def resolution_messages(subject: str, body: str, policy_ids: list[str]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": RESOLUTION_SYSTEM},
        {"role": "user", "content": resolution_user_prompt(subject, body, policy_ids)},
    ]
