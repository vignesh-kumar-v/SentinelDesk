"""The guardrails layer: NeMo Guardrails for judgement, deterministic checks for facts.

Split by what kind of question each check answers:

* **PII** (`pii.py`) is regex. It is auditable, costs microseconds, cannot be
  prompt-injected, and has better recall on structured identifiers than any model.
* **Topic and jailbreak rails** are NeMo Guardrails, because "is this off-topic" and
  "is this trying to override the agent's instructions" are judgement calls with
  fuzzy boundaries, which is what an LLM rail is for.
* **Schema** (`schema.py`) is pydantic. Structure is not a judgement call either.

The rail model is deliberately not the resolution model. A 0.5B model fine-tuned on
support responses, asked to police its own output, is a checker that shares every
blind spot of the thing it checks.

Rails run in "rails-only" mode: NeMo checks the text and never generates the reply.
Generation belongs to the DPO-tuned model, which is the point of the project.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ..config import Paths
from ..logging_utils import get_logger
from .pii import redact

log = get_logger(__name__)

DEFAULT_CONFIG = Paths.configs / "guardrails"

# NeMo's refusal text when a rail fires, from configs/guardrails/config.yml.
_BLOCK_MARKERS = ("i'm sorry, i can't respond to that", "i can only help with nimbus")


@dataclass
class GuardDecision:
    stage: Literal["input", "output"]
    allowed: bool
    text: str
    original: str
    reasons: list[str] = field(default_factory=list)
    pii_found: dict[str, int] = field(default_factory=dict)
    rail_latency_s: float = 0.0
    rail_available: bool = True

    @property
    def modified(self) -> bool:
        return self.text != self.original


class Guardrails:
    """Input and output rails around the resolution agent."""

    def __init__(
        self,
        config_path: Path | str = DEFAULT_CONFIG,
        *,
        use_llm_rails: bool = True,
        redact_pii: bool = True,
    ) -> None:
        self.redact_pii = redact_pii
        self.rails = None
        self._options = None
        if not use_llm_rails:
            return
        try:
            from nemoguardrails import LLMRails, RailsConfig
            from nemoguardrails.rails.llm.options import GenerationOptions

            self.rails = LLMRails(RailsConfig.from_path(str(config_path)))
            self._options = GenerationOptions
            log.info("NeMo Guardrails loaded from %s", config_path)
        except Exception as exc:  # noqa: BLE001
            # A dead rail model must not take the support pipeline down with it. The
            # deterministic rails still run, and `rail_available=False` is recorded so
            # the degradation is visible rather than silently permissive.
            log.warning("NeMo Guardrails unavailable (%s); deterministic rails only", exc)

    # ------------------------------------------------------------------ internals
    def _run_rail(self, messages: list[dict[str, str]], stage: str) -> tuple[bool, str, float]:
        """Returns (allowed, rail_message, latency). Never raises."""
        if self.rails is None or self._options is None:
            return True, "", 0.0
        t0 = time.perf_counter()
        try:
            result = self.rails.generate(
                messages=messages, options=self._options(rails=[stage])
            )
            payload = result.response
            content = (
                payload[-1].get("content", "") if isinstance(payload, list) and payload
                else str(payload)
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("%s rail failed (%s); allowing through", stage, type(exc).__name__)
            return True, "", time.perf_counter() - t0
        lowered = content.strip().lower()
        blocked = any(m in lowered for m in _BLOCK_MARKERS)
        return (not blocked), content.strip(), time.perf_counter() - t0

    # ------------------------------------------------------------------ public
    def check_input(self, text: str) -> GuardDecision:
        reasons: list[str] = []
        out = text
        found: dict[str, int] = {}

        if self.redact_pii:
            r = redact(text)
            if r.redacted:
                out, found = r.text, r.found
                reasons.append(f"redacted inbound PII: {', '.join(r.labels)}")

        # The rail sees the redacted text: there is no reason to ship a customer's card
        # number to the rail provider just to be told the message is on-topic.
        allowed, rail_msg, latency = self._run_rail(
            [{"role": "user", "content": out}], "input"
        )
        if not allowed:
            reasons.append("input rail: off-topic, unsafe, or an instruction override")

        return GuardDecision(
            stage="input", allowed=allowed, text=out, original=text, reasons=reasons,
            pii_found=found, rail_latency_s=round(latency, 4),
            rail_available=self.rails is not None,
        )

    def check_output(self, text: str, *, user_text: str = "") -> GuardDecision:
        reasons: list[str] = []
        out = text
        found: dict[str, int] = {}

        if self.redact_pii:
            r = redact(text)
            if r.redacted:
                out, found = r.text, r.found
                # Outbound PII is the more serious direction: the model repeating an
                # identifier back can leak it into logs, tickets and email threads.
                reasons.append(f"redacted outbound PII: {', '.join(r.labels)}")

        allowed, rail_msg, latency = self._run_rail(
            [{"role": "user", "content": user_text or "customer ticket"},
             {"role": "assistant", "content": out}],
            "output",
        )
        if not allowed:
            reasons.append("output rail: unsafe, out-of-scope, or exceeds support authority")

        return GuardDecision(
            stage="output", allowed=allowed, text=out, original=text, reasons=reasons,
            pii_found=found, rail_latency_s=round(latency, 4),
            rail_available=self.rails is not None,
        )
