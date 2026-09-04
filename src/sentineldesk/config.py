"""Central configuration.

Every path in the project is derived from ROOT here rather than from the caller's
working directory, so `sentineldesk ...` behaves the same from any cwd.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Paths:
    root = ROOT
    data = ROOT / "data"
    raw = ROOT / "data" / "raw"
    processed = ROOT / "data" / "processed"
    prefs = ROOT / "data" / "prefs"
    artifacts = ROOT / "artifacts"
    reports = ROOT / "reports"
    configs = ROOT / "configs"

    @classmethod
    def ensure(cls) -> None:
        for p in (cls.data, cls.raw, cls.processed, cls.prefs, cls.artifacts, cls.reports):
            p.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """Runtime settings. Env vars take the `SD_` prefix; see .env.example."""

    model_config = SettingsConfigDict(
        env_prefix="SD_", env_file=ROOT / ".env", extra="ignore", protected_namespaces=()
    )

    # --- judge (frontier model, used for preference labels and the win-rate arena)
    judge_base_url: str = "http://localhost:11434/v1"
    judge_api_key: str = "ollama"
    judge_model: str = "deepseek-v4-pro:cloud"
    judge_temperature: float = 0.0
    judge_max_retries: int = 4
    judge_max_tokens: int = 12000
    judge_concurrency: int = 4

    # --- candidate generation (the model whose outputs become preference pairs)
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    gen_base_url: str = "http://localhost:11434/v1"
    gen_api_key: str = "ollama"

    # --- resolution agent serving endpoint (vLLM from Phase 4 onward)
    resolution_base_url: str = "http://localhost:8000/v1"
    resolution_api_key: str = "EMPTY"
    resolution_model: str = "sentineldesk-dpo"

    # --- reproducibility
    seed: int = 1337

    # --- escalation rule (deliberately rule-based; see docs/decisions.md)
    escalation_confidence_threshold: float = 0.55

    device: str = Field(default="auto", description="auto | mps | cuda | cpu")

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
