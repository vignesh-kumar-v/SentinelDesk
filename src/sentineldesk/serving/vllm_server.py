"""Launching and health-checking a vLLM OpenAI-compatible server.

vLLM ships no macOS wheel and has no Metal backend, so on this machine it runs as a
from-source CPU build in its own virtualenv (scripts/build_vllm_macos.sh). That is
why the server is started as a subprocess against a specific interpreter rather than
imported: the serving environment pins torch 2.8.0 CPU, and the training environment
must not.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from ..config import Paths
from ..logging_utils import get_logger

log = get_logger(__name__)

DEFAULT_VLLM_PYTHON = Paths.root / ".venv-vllm" / "bin" / "python"


@dataclass
class VLLMServer:
    model_path: str
    served_name: str = "sentineldesk-dpo"
    port: int = 8000
    host: str = "127.0.0.1"
    python: Path = DEFAULT_VLLM_PYTHON
    max_model_len: int = 2048
    dtype: str = "bfloat16"
    extra_args: tuple[str, ...] = ()
    log_path: Path | None = None
    _proc: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def _cmd(self) -> list[str]:
        return [
            str(self.python), "-m", "vllm.entrypoints.openai.api_server",
            "--model", str(self.model_path),
            "--served-model-name", self.served_name,
            "--host", self.host,
            "--port", str(self.port),
            "--max-model-len", str(self.max_model_len),
            "--dtype", self.dtype,
            *self.extra_args,
        ]

    def start(self, *, timeout_s: float = 900.0) -> None:
        if not self.python.exists():
            raise FileNotFoundError(
                f"no vLLM interpreter at {self.python}; run scripts/build_vllm_macos.sh first"
            )
        if self.is_up():
            log.info("vLLM already serving on %s", self.base_url)
            return

        log_path = self.log_path or Paths.reports / "vllm_server.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env["VLLM_TARGET_DEVICE"] = "cpu"
        env.setdefault("VLLM_LOGGING_LEVEL", "INFO")
        log.info("starting vLLM: %s", " ".join(self._cmd()))
        with log_path.open("w") as fh:
            self._proc = subprocess.Popen(
                self._cmd(), stdout=fh, stderr=subprocess.STDOUT, env=env,
                # own process group so stop() can take down vLLM's worker children too
                start_new_session=True,
            )
        self.wait_until_up(timeout_s=timeout_s, log_path=log_path)

    def is_up(self) -> bool:
        try:
            r = httpx.get(f"http://{self.host}:{self.port}/health", timeout=2.0)
            return r.status_code == 200
        except Exception:  # noqa: BLE001
            return False

    def wait_until_up(self, *, timeout_s: float, log_path: Path | None = None) -> None:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if self.is_up():
                log.info("vLLM up after %.0fs on %s", time.time() - t0, self.base_url)
                return
            if self._proc is not None and self._proc.poll() is not None:
                tail = ""
                if log_path and log_path.exists():
                    tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-30:])
                raise RuntimeError(
                    f"vLLM exited with code {self._proc.returncode} before serving.\n{tail}"
                )
            time.sleep(3.0)
        raise TimeoutError(f"vLLM did not come up within {timeout_s:.0f}s; see {log_path}")

    def stop(self) -> None:
        if self._proc is None or self._proc.poll() is not None:
            return
        os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
        try:
            self._proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
        log.info("vLLM stopped")

    def __enter__(self) -> VLLMServer:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
