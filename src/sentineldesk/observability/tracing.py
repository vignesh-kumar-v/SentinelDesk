"""Langfuse tracing for the agent graph.

Two properties this is built around.

**The pipeline must run identically without it.** Observability that can take the
system down is worse than none, so a missing SDK, missing credentials or an
unreachable server all degrade to a no-op tracer. `enabled` records which happened, so
"no traces" is distinguishable from "traces are fine, nobody looked".

**Each hop is typed, not a generic span.** Langfuse's semantic observation types map
onto this graph almost exactly — the retrieval step is a `retriever`, the rails are
`guardrail`s, the resolution call is a `generation` that carries model name and token
counts. A trace of seven identical spans tells you the order things happened; a typed
trace tells you a guardrail fired on a retrieval that returned nothing, which is the
question you actually open a trace to answer.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from ..logging_utils import get_logger

log = get_logger(__name__)

# node name -> Langfuse observation type
NODE_TYPES: dict[str, str] = {
    "guard_input": "guardrail",
    "triage": "chain",
    "retrieve": "retriever",
    "resolution": "generation",
    "guard_output": "guardrail",
    "gate": "span",
    "respond": "span",
    "escalate": "span",
}


@dataclass
class Tracer:
    """Wraps Langfuse. Safe to construct and use when Langfuse is absent."""

    enabled: bool = False
    reason: str = "not configured"
    client: Any = None
    _root: Any = field(default=None, repr=False)

    @classmethod
    def create(cls, *, force_disable: bool = False) -> Tracer:
        if force_disable:
            return cls(enabled=False, reason="disabled by caller")
        if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
            return cls(enabled=False, reason="LANGFUSE_PUBLIC_KEY/SECRET_KEY not set")
        try:
            from langfuse import Langfuse

            client = Langfuse(
                public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
                secret_key=os.environ["LANGFUSE_SECRET_KEY"],
                host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            )
            return cls(enabled=True, reason="ok", client=client)
        except Exception as exc:  # noqa: BLE001
            return cls(enabled=False, reason=f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ spans
    @contextmanager
    def ticket(self, ticket_id: str, subject: str, body: str, **meta: Any):
        """Root span for one ticket's journey through the graph."""
        if not self.enabled:
            yield self
            return
        with self.client.start_as_current_observation(
            name=f"ticket:{ticket_id}", as_type="agent",
            input={"subject": subject, "body": body},
            metadata={"ticket_id": ticket_id, **meta},
        ) as root:
            self._root = root
            try:
                yield self
            finally:
                self._root = None

    @contextmanager
    def node(self, name: str, **input_kwargs: Any):
        """One graph node. Yields a handle whose .finish() records the output."""
        if not self.enabled:
            yield _NullSpan()
            return
        with self.client.start_as_current_observation(
            name=name, as_type=NODE_TYPES.get(name, "span"), input=input_kwargs or None
        ) as span:
            yield _Span(span)

    def update_ticket(self, **kwargs: Any) -> None:
        if self.enabled and self._root is not None:
            try:
                self._root.update(**kwargs)
            except Exception as exc:  # noqa: BLE001
                log.debug("trace update failed: %s", exc)

    def score(self, name: str, value: float, comment: str = "") -> None:
        """Attach a numeric score to the current trace (confidence, judge rating)."""
        if not self.enabled or self._root is None:
            return
        try:
            self._root.score(name=name, value=value, comment=comment or None)
        except Exception as exc:  # noqa: BLE001
            log.debug("scoring failed: %s", exc)

    def flush(self) -> None:
        if self.enabled and self.client is not None:
            try:
                self.client.flush()
            except Exception as exc:  # noqa: BLE001
                log.warning("langfuse flush failed: %s", exc)


class _NullSpan:
    def finish(self, **kwargs: Any) -> None:
        return

    def set_usage(self, **kwargs: Any) -> None:
        return


@dataclass
class _Span:
    span: Any

    def finish(self, output: Any = None, **metadata: Any) -> None:
        try:
            payload: dict[str, Any] = {}
            if output is not None:
                payload["output"] = output
            if metadata:
                payload["metadata"] = metadata
            if payload:
                self.span.update(**payload)
        except Exception as exc:  # noqa: BLE001
            log.debug("span finish failed: %s", exc)

    def set_usage(self, *, model: str = "", input_tokens: int = 0, output_tokens: int = 0) -> None:
        """Record generation model and token counts so cost rolls up in the UI."""
        try:
            self.span.update(
                model=model or None,
                usage_details={"input": input_tokens, "output": output_tokens,
                               "total": input_tokens + output_tokens},
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("usage update failed: %s", exc)
