"""Launching and health-checking a vLLM OpenAI-compatible server.

Two backends, because on this hardware only one of them actually works:

* ``docker`` (default) runs vLLM's CPU backend built for linux/arm64. This is the
  path that serves.
* ``native`` runs the from-source macOS CPU build in ``.venv-vllm``. It compiles and
  installs, then deadlocks at 0% CPU during CPU-worker init. It is kept because the
  build script and the failure are part of the project's record, and because the
  same code path is what would run on a Linux host with no container.

See docs/decisions.md F1 and F2. Both backends expose the same OpenAI-compatible
endpoint, so nothing above this module knows which one is running.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from ..config import Paths
from ..logging_utils import get_logger

log = get_logger(__name__)

DEFAULT_VLLM_PYTHON = Paths.root / ".venv-vllm" / "bin" / "python"
DEFAULT_IMAGE = "sentineldesk/vllm-cpu:0.11.0-pinned"
CONTAINER_NAME = "sentineldesk-vllm"


@dataclass
class VLLMServer:
    model_path: str
    served_name: str = "sentineldesk-dpo"
    port: int = 8000
    host: str = "127.0.0.1"
    backend: str = "docker"
    image: str = DEFAULT_IMAGE
    container_name: str = CONTAINER_NAME
    python: Path = DEFAULT_VLLM_PYTHON
    max_model_len: int = 2048
    dtype: str = "bfloat16"
    kv_cache_gb: int = 4
    memory: str = "6g"
    enforce_eager: bool = True
    extra_args: tuple[str, ...] = ()
    log_path: Path | None = None
    _proc: subprocess.Popen | None = field(default=None, repr=False)
    _started_container: bool = field(default=False, repr=False)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    # ------------------------------------------------------------------ common
    def _server_args(self, model: str) -> list[str]:
        args = [
            "--model", model,
            "--served-model-name", self.served_name,
            "--max-model-len", str(self.max_model_len),
            "--dtype", self.dtype,
        ]
        if self.enforce_eager:
            args.append("--enforce-eager")
        return args + list(self.extra_args)

    def is_up(self) -> bool:
        try:
            return httpx.get(f"http://{self.host}:{self.port}/health", timeout=2.0).status_code == 200
        except Exception:  # noqa: BLE001
            return False

    def start(self, *, timeout_s: float = 1800.0) -> None:
        if self.is_up():
            log.info("vLLM already serving on %s", self.base_url)
            return
        if self.backend == "docker":
            self._start_docker()
        elif self.backend == "native":
            self._start_native()
        else:
            raise ValueError(f"unknown vLLM backend {self.backend!r}")
        self.wait_until_up(timeout_s=timeout_s)

    def wait_until_up(self, *, timeout_s: float) -> None:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if self.is_up():
                log.info("vLLM up after %.0fs on %s", time.time() - t0, self.base_url)
                return
            if not self._still_alive():
                raise RuntimeError(f"vLLM exited before serving.\n{self.recent_logs()}")
            time.sleep(3.0)
        raise TimeoutError(f"vLLM did not come up within {timeout_s:.0f}s.\n{self.recent_logs()}")

    # ------------------------------------------------------------------ docker
    def _docker(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(["docker", *args], capture_output=True, text=True, check=check)

    def _start_docker(self) -> None:
        if shutil.which("docker") is None:
            raise FileNotFoundError("docker not found on PATH")
        if self._docker("info", check=False).returncode != 0:
            raise RuntimeError("the docker daemon is not running")
        if not self._docker("image", "inspect", self.image, check=False).returncode == 0:
            raise FileNotFoundError(
                f"image {self.image} is missing; build it with scripts/build_vllm_docker.sh "
                "then docker/Dockerfile.vllm-pins"
            )

        self._docker("rm", "-f", self.container_name, check=False)

        local = Path(self.model_path)
        mounts: list[str] = []
        if local.exists():
            # A local checkpoint has to be visible inside the container. Mounted
            # read-only: the server has no business writing to a training artifact.
            mounts += ["-v", f"{local.resolve()}:/model:ro"]
            model_arg = "/model"
        else:
            model_arg = self.model_path
        # Share the host HF cache either way, so a hub model is not re-downloaded per run.
        hf_cache = Path.home() / ".cache" / "huggingface"
        hf_cache.mkdir(parents=True, exist_ok=True)
        mounts += ["-v", f"{hf_cache}:/root/.cache/huggingface"]

        cmd = [
            "run", "-d", "--name", self.container_name,
            "-p", f"{self.port}:8000",
            "-e", f"VLLM_CPU_KVCACHE_SPACE={self.kv_cache_gb}",
            "--memory", self.memory,
            *mounts,
            self.image,
            *self._server_args(model_arg),
        ]
        log.info("starting vLLM container: docker %s", " ".join(cmd))
        self._docker(*cmd)
        self._started_container = True

    # ------------------------------------------------------------------ native
    def _start_native(self) -> None:
        if not self.python.exists():
            raise FileNotFoundError(
                f"no vLLM interpreter at {self.python}; run scripts/build_vllm_macos.sh first"
            )
        log_path = self.log_path or Paths.reports / "vllm_server.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env["VLLM_TARGET_DEVICE"] = "cpu"
        env["VLLM_CPU_KVCACHE_SPACE"] = str(self.kv_cache_gb)
        cmd = [
            str(self.python), "-m", "vllm.entrypoints.openai.api_server",
            "--host", self.host, "--port", str(self.port),
            *self._server_args(self.model_path),
        ]
        log.warning(
            "starting the native macOS vLLM build; it is known to hang at CPU-worker "
            "init on this platform (docs/decisions.md F1)"
        )
        with log_path.open("w") as fh:
            self._proc = subprocess.Popen(
                cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, start_new_session=True
            )
        self.log_path = log_path

    # ------------------------------------------------------------------ status
    def _still_alive(self) -> bool:
        if self.backend == "docker":
            r = self._docker("inspect", "-f", "{{.State.Running}}", self.container_name, check=False)
            return r.stdout.strip() == "true"
        return self._proc is None or self._proc.poll() is None

    def recent_logs(self, n: int = 30) -> str:
        if self.backend == "docker":
            r = self._docker("logs", "--tail", str(n), self.container_name, check=False)
            return (r.stdout + r.stderr)[-4000:]
        if self.log_path and self.log_path.exists():
            return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-n:])
        return "(no logs)"

    def stop(self) -> None:
        if self.backend == "docker":
            if self._started_container:
                self._docker("rm", "-f", self.container_name, check=False)
                log.info("vLLM container removed")
            return
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
