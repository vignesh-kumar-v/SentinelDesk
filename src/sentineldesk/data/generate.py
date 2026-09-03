"""Synthetic support tickets, each grounded in a ground-truth policy.

Every ticket is written *from* a policy in configs/policies.yaml, so the correct
answer is a known thing rather than the judge's opinion. Roughly a third of tickets
carry a wrong customer assumption the policy contradicts; those are where response
quality actually separates, because a weak model happily agrees with the customer.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass

from ..config import get_settings
from ..llm import ChatClient, LLMError, extract_json
from ..logging_utils import get_logger
from .kb import load_policies
from .schema import Policy, Ticket

log = get_logger(__name__)

PERSONAS = [
    "a terse engineer who writes in fragments",
    "a frustrated small-business owner using lots of capitals",
    "a polite first-time customer who over-explains",
    "an IT admin managing 40 seats, writes formally",
    "a non-native English speaker, simple sentences, some grammar slips",
    "a customer replying to a previous ticket, references a ticket number",
    "someone typing quickly on a phone, no punctuation, some typos",
    "a procurement manager asking on behalf of a colleague",
    "a developer who pastes an error code and little else",
    "a long-time customer who mentions how long they've been with us",
]

TWISTS = [
    ("plain", "The customer simply asks the question."),
    (
        "wrong_assumption",
        "The customer states a confident but WRONG assumption that the policy contradicts. "
        "Do not signal that it is wrong - write it the way a real customer would state it.",
    ),
    (
        "wrong_assumption",
        "The customer claims a colleague or a support agent already promised them something the "
        "policy does not allow. Do not flag it as wrong.",
    ),
    (
        "urgent_pressure",
        "The customer applies deadline pressure or threatens to churn, but the underlying "
        "question is still answered by the policy.",
    ),
    (
        "multi_part",
        "The customer asks the main question plus one small unrelated side question.",
    ),
    (
        "vague",
        "The customer describes the symptom without naming the feature, so the agent has to "
        "work out which policy applies.",
    ),
]

_SYSTEM = (
    "You write realistic inbound customer-support tickets for a company called Nimbus "
    "(a SaaS product with some hardware accessories). You output only JSON."
)

_TEMPLATE = """Write ONE realistic customer support ticket.

The ticket must be answerable by this internal policy (the customer does NOT know the policy text):
[{pid}] {ptitle}
{pbody}

Persona: {persona}
Scenario: {twist}
Urgency to convey: {urgency}

Rules:
- Never quote or paraphrase the policy text back. The customer does not have it.
- Do not include the answer. The ticket is the question only.
- 25-90 words in the body. Write in the persona's voice, including its flaws.
- Invent plausible specifics (dates, order ids, plan names, amounts) where natural.

Return only this JSON:
{{"subject": "...", "body": "..."}}"""


@dataclass
class TicketGenerator:
    client: ChatClient
    model: str
    seed: int = 1337

    def _plan(self, n: int) -> list[tuple[Policy, str, tuple[str, str], str]]:
        """Deterministic assignment of policy x persona x twist x urgency.

        Policies are cycled rather than sampled so every policy gets near-equal
        coverage; with 22 policies and a few hundred tickets, sampling leaves some
        policies with two tickets and others with fifteen.
        """
        rng = random.Random(self.seed)
        policies = load_policies()
        plan = []
        for i in range(n):
            pol = policies[i % len(policies)]
            persona = PERSONAS[(i // len(policies) + i) % len(PERSONAS)]
            twist = TWISTS[rng.randrange(len(TWISTS))]
            urgency = rng.choices(["low", "medium", "high"], weights=[0.4, 0.4, 0.2])[0]
            plan.append((pol, persona, twist, urgency))
        return plan

    def generate(self, n: int, concurrency: int = 6) -> list[Ticket]:
        plan = self._plan(n)
        prompts = [
            [
                {"role": "system", "content": _SYSTEM},
                {
                    "role": "user",
                    "content": _TEMPLATE.format(
                        pid=pol.id,
                        ptitle=pol.title,
                        pbody=" ".join(pol.body.split()),
                        persona=persona,
                        twist=twist[1],
                        urgency=urgency,
                    ),
                },
            ]
            for pol, persona, twist, urgency in plan
        ]

        log.info("generating %d tickets via %s (concurrency=%d)", n, self.model, concurrency)
        results = self.client.map_chat(
            prompts,
            concurrency=concurrency,
            model=self.model,
            # High temperature on purpose: at temperature 0 this prompt collapses to
            # near-identical tickets per policy, and 22 distinct tickets is not a dataset.
            temperature=1.0,
            max_tokens=2048,
        )

        tickets: list[Ticket] = []
        dropped = 0
        seen_bodies: set[str] = set()
        for (pol, _persona, twist, urgency), res in zip(plan, results, strict=True):
            obj = extract_json(res.text)
            if not obj or not obj.get("subject") or not obj.get("body"):
                dropped += 1
                continue
            body = str(obj["body"]).strip()
            fingerprint = hashlib.sha1(body.lower().encode()).hexdigest()
            if fingerprint in seen_bodies:
                dropped += 1
                continue
            seen_bodies.add(fingerprint)
            tickets.append(
                Ticket(
                    id=f"T{len(tickets):04d}",
                    category=pol.category,
                    urgency=urgency,  # type: ignore[arg-type]
                    subject=str(obj["subject"]).strip()[:160],
                    body=body,
                    policy_ids=[pol.id],
                    source=f"synthetic:{twist[0]}:{self.model}",
                )
            )
        if dropped:
            log.warning("dropped %d generations (unparseable or duplicate)", dropped)
        return tickets


def split_tickets(tickets: list[Ticket], heldout_frac: float, seed: int) -> list[Ticket]:
    """Stratified train/heldout split.

    Stratified by (category, twist) rather than random: the held-out set is what the
    Phase 5 win-rate is measured on, and a random split can hand it an unrepresentative
    mix of easy plain tickets, which would flatter or bury the result for no real reason.
    """
    rng = random.Random(seed)
    strata: dict[tuple[str, str], list[Ticket]] = {}
    for t in tickets:
        twist = t.source.split(":")[1] if ":" in t.source else "plain"
        strata.setdefault((t.category, twist), []).append(t)

    for group in strata.values():
        rng.shuffle(group)
        n_held = round(len(group) * heldout_frac)
        for i, t in enumerate(group):
            t.split = "heldout" if i < n_held else "train"
    return tickets


def coverage_report(tickets: list[Ticket]) -> dict:
    from collections import Counter

    twists = Counter(t.source.split(":")[1] if ":" in t.source else "plain" for t in tickets)
    return {
        "n": len(tickets),
        "by_category": dict(Counter(t.category for t in tickets)),
        "by_urgency": dict(Counter(t.urgency for t in tickets)),
        "by_twist": dict(twists),
        "by_split": dict(Counter(t.split for t in tickets)),
        "policies_covered": len({p for t in tickets for p in t.policy_ids}),
        "median_body_words": sorted(len(t.body.split()) for t in tickets)[len(tickets) // 2]
        if tickets
        else 0,
    }


def build_generator() -> TicketGenerator:
    s = get_settings()
    return TicketGenerator(
        client=ChatClient(s.judge_base_url, s.judge_api_key, s.judge_model, max_retries=s.judge_max_retries),
        model=s.judge_model,
        seed=s.seed,
    )


__all__ = ["TicketGenerator", "build_generator", "split_tickets", "coverage_report", "LLMError", "json"]
