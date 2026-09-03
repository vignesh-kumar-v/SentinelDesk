"""The resolution node: the one node whose model is a training artifact.

Two backends behind one interface. `RemoteResolver` speaks the OpenAI protocol and
is what Phase 4 points at vLLM; `LocalResolver` runs the checkpoint in-process with
transformers and exists so the graph is runnable, and testable, without a server up.
Both return the same (text, metadata) shape, so swapping them changes nothing above
this line.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from ..llm import ChatClient, LLMError
from ..logging_utils import get_logger
from ..prompts import RESOLUTION_SYSTEM, resolution_user_prompt
from .state import TicketState

log = get_logger(__name__)


class Resolver(Protocol):
    name: str

    def resolve(self, messages: list[dict[str, str]]) -> tuple[str, dict]: ...


@dataclass
class RemoteResolver:
    """Calls an OpenAI-compatible server (vLLM in Phase 4)."""

    client: ChatClient
    model: str
    temperature: float = 0.3
    max_tokens: int = 300
    request_logprobs: bool = True
    name: str = "remote"

    def resolve(self, messages: list[dict[str, str]]) -> tuple[str, dict]:
        extra = {"logprobs": True, "top_logprobs": 0} if self.request_logprobs else None
        try:
            res = self.client.chat(
                messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                extra_body=extra,
            )
        except LLMError:
            if not self.request_logprobs:
                raise
            # Not every OpenAI-compatible server accepts logprobs on chat completions.
            # Losing one confidence signal is a far better outcome than a dead node.
            log.warning("resolution call rejected logprobs; retrying without")
            self.request_logprobs = False
            res = self.client.chat(
                messages, model=self.model, temperature=self.temperature, max_tokens=self.max_tokens
            )
        return res.text, {
            "model": res.model,
            "latency_s": res.latency_s,
            "tokens": res.completion_tokens,
            "mean_logprob": None,
        }


@dataclass
class LocalResolver:
    """Runs a checkpoint in-process. Used by tests and by the pre-vLLM graph."""

    generator: object  # serving.local.HFGenerator
    temperature: float = 0.3
    max_tokens: int = 300
    name: str = "local"

    def resolve(self, messages: list[dict[str, str]]) -> tuple[str, dict]:
        t0 = time.perf_counter()
        out = self.generator.generate(  # type: ignore[attr-defined]
            [messages],
            temperature=self.temperature,
            top_p=0.9,
            max_tokens=self.max_tokens,
            show_progress=False,
        )[0]
        stats = getattr(self.generator, "last_stats", {})
        return out.strip(), {
            "model": "local",
            "latency_s": time.perf_counter() - t0,
            "tokens": int(stats.get("output_tokens", 0)),
            "mean_logprob": None,
        }


def make_retrieve_node():
    """Fetch the policy excerpts for the triaged category."""
    from ..data.kb import retrieve_for_ticket

    def retrieve(state: TicketState) -> dict:
        t0 = time.perf_counter()
        ids = retrieve_for_ticket(state["category"])  # type: ignore[arg-type]
        return {
            "policy_ids": ids,
            "trace": [
                {"node": "retrieve", "latency_s": round(time.perf_counter() - t0, 4), "n_policies": len(ids)}
            ],
        }

    return retrieve


def make_resolution_node(resolver: Resolver):
    def resolution(state: TicketState) -> dict:
        t0 = time.perf_counter()
        messages = [
            {"role": "system", "content": RESOLUTION_SYSTEM},
            {
                "role": "user",
                "content": resolution_user_prompt(
                    state.get("subject", ""), state.get("body", ""), state.get("policy_ids", [])
                ),
            },
        ]
        try:
            text, meta = resolver.resolve(messages)
        except Exception as exc:  # noqa: BLE001 - the graph must still route the ticket
            log.error("resolution failed for %s: %s", state.get("ticket_id"), exc)
            text, meta = "", {"model": "error", "latency_s": 0.0, "tokens": 0, "mean_logprob": None,
                              "error": str(exc)[:200]}
        return {
            "draft": text,
            "resolution_model": meta.get("model", ""),
            "resolution_latency_s": round(float(meta.get("latency_s", 0.0)), 4),
            "resolution_tokens": int(meta.get("tokens", 0)),
            "mean_logprob": meta.get("mean_logprob"),
            "trace": [
                {
                    "node": "resolution",
                    "latency_s": round(time.perf_counter() - t0, 4),
                    "backend": resolver.name,
                    "model": meta.get("model", ""),
                    "tokens": meta.get("tokens", 0),
                    "error": meta.get("error"),
                }
            ],
        }

    return resolution
