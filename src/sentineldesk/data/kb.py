"""The support knowledge base: ground-truth policies loaded from configs/policies.yaml."""

from __future__ import annotations

from functools import lru_cache

import yaml

from ..config import Paths
from .schema import CATEGORIES, Category, Policy

KB_PATH = Paths.configs / "policies.yaml"


@lru_cache(maxsize=1)
def load_policies() -> list[Policy]:
    rows = yaml.safe_load(KB_PATH.read_text(encoding="utf-8"))
    policies = [Policy.model_validate(r) for r in rows]
    ids = [p.id for p in policies]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate policy ids in configs/policies.yaml")
    missing = set(CATEGORIES) - {p.category for p in policies}
    if missing:
        raise ValueError(f"no policies for categories: {sorted(missing)}")
    return policies


@lru_cache(maxsize=1)
def _by_id() -> dict[str, Policy]:
    return {p.id: p for p in load_policies()}


def get_policy(policy_id: str) -> Policy:
    try:
        return _by_id()[policy_id]
    except KeyError as exc:
        raise KeyError(f"unknown policy id {policy_id!r}") from exc


def policies_for_category(category: Category) -> list[Policy]:
    return [p for p in load_policies() if p.category == category]


def render_policies(policy_ids: list[str]) -> str:
    """Format policies as the context block an agent or judge sees."""
    if not policy_ids:
        return "(no policy excerpts available)"
    parts = []
    for pid in policy_ids:
        p = get_policy(pid)
        parts.append(f"[{p.id}] {p.title}\n{' '.join(p.body.split())}")
    return "\n\n".join(parts)


def retrieve_for_ticket(category: Category, policy_ids: list[str] | None = None) -> list[str]:
    """Context the resolution agent gets at inference time.

    Deliberately category-level rather than the exact policy the ticket was written
    from: handing the agent precisely the one right policy would make the task
    trivial and the preference signal meaningless. It has to pick the relevant one
    out of the category, which is where responses actually differ in correctness.
    """
    cat_ids = [p.id for p in policies_for_category(category)]
    if policy_ids:
        # keep the ticket's own policies first, then the rest of the category
        ordered = [pid for pid in policy_ids if pid in cat_ids]
        ordered += [pid for pid in cat_ids if pid not in ordered]
        return ordered
    return cat_ids
