"""One OpenAI-compatible chat client for every model this project talks to.

The judge, the candidate generator, the triage node and the vLLM-served resolution
agent all speak the same wire protocol, so they share this. Keeping it in one place
is what makes the Phase 4 swap (Ollama -> vLLM) a config change rather than a rewrite.
"""

from __future__ import annotations

import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import httpx

from .logging_utils import get_logger

log = get_logger(__name__)

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class LLMError(RuntimeError):
    pass


@dataclass
class ChatResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    model: str = ""
    # Reasoning models (deepseek-v4, kimi, glm) return this alongside content.
    # We keep it for trace/debug but never feed it back into a prompt.
    reasoning: str = ""


@dataclass
class ChatClient:
    base_url: str
    api_key: str = "EMPTY"
    model: str = ""
    timeout_s: float = 300.0
    max_retries: int = 4
    _client: httpx.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = httpx.Client(
            base_url=self.base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=httpx.Timeout(self.timeout_s, connect=15.0),
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> ChatClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ chat
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int | None = None,
        stop: list[str] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if seed is not None:
            payload["seed"] = seed
        if stop:
            payload["stop"] = stop
        if extra_body:
            payload.update(extra_body)

        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            t0 = time.perf_counter()
            try:
                resp = self._client.post("/chat/completions", json=payload)
                if resp.status_code >= 500 or resp.status_code == 429:
                    raise LLMError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                resp.raise_for_status()
                data = resp.json()
                msg = data["choices"][0]["message"]
                usage = data.get("usage") or {}
                return ChatResult(
                    text=(msg.get("content") or "").strip(),
                    prompt_tokens=int(usage.get("prompt_tokens") or 0),
                    completion_tokens=int(usage.get("completion_tokens") or 0),
                    latency_s=time.perf_counter() - t0,
                    model=data.get("model", payload["model"]),
                    reasoning=(msg.get("reasoning") or "").strip(),
                )
            except Exception as exc:  # noqa: BLE001 - retry on anything transport-ish
                last_exc = exc
                # Full jitter: a burst of concurrent workers that all 429 at once must
                # not retry in lockstep, or they just recreate the same burst.
                backoff = min(2**attempt, 16) * random.random()
                log.warning(
                    "chat attempt %d/%d failed (%s); retrying in %.1fs",
                    attempt + 1,
                    self.max_retries,
                    type(exc).__name__,
                    backoff,
                )
                time.sleep(backoff)
        raise LLMError(f"chat failed after {self.max_retries} attempts: {last_exc}") from last_exc

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        required_keys: tuple[str, ...] = (),
        **kwargs: Any,
    ) -> tuple[dict[str, Any], ChatResult]:
        """Chat, then parse the reply as JSON, retrying the *model* on bad JSON.

        A retry here re-rolls the sampler, so it is only useful with a nonzero
        temperature on the retry - at temperature 0 a model that emitted broken JSON
        will emit exactly the same broken JSON forever.
        """
        attempts = kwargs.pop("json_retries", 3)
        base_temp = kwargs.pop("temperature", 0.0)
        last: ChatResult | None = None
        for i in range(attempts):
            res = self.chat(messages, temperature=base_temp if i == 0 else 0.4, **kwargs)
            last = res
            obj = extract_json(res.text)
            if obj is not None and all(k in obj for k in required_keys):
                return obj, res
            log.warning("unparseable/incomplete JSON on attempt %d: %r", i + 1, res.text[:160])
        raise LLMError(f"no valid JSON after {attempts} attempts; last reply: {last.text[:300] if last else ''}")

    # ------------------------------------------------------------------ batch
    def map_chat(
        self,
        batches: list[list[dict[str, str]]],
        *,
        concurrency: int = 4,
        **kwargs: Any,
    ) -> list[ChatResult]:
        """Run many chats concurrently, preserving input order."""
        if concurrency <= 1:
            return [self.chat(m, **kwargs) for m in batches]
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            return list(pool.map(lambda m: self.chat(m, **kwargs), batches))


def extract_json(text: str) -> dict[str, Any] | None:
    """Pull a JSON object out of a model reply.

    Models wrap JSON in prose or fences often enough that a bare json.loads throws
    away perfectly good verdicts, so try progressively looser strategies.
    """
    text = text.strip()
    for candidate in _json_candidates(text):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _json_candidates(text: str) -> list[str]:
    out = [text]
    fenced = _JSON_FENCE.search(text)
    if fenced:
        out.append(fenced.group(1))
    first, last = text.find("{"), text.rfind("}")
    if first != -1 and last > first:
        out.append(text[first : last + 1])
    return out
