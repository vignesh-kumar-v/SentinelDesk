"""Phase 4: serving throughput and latency, measured the way vLLM is actually used.

Two regimes, because reporting only one misrepresents what a serving engine does:

* Sequential - one request at a time. This is per-ticket latency, what a single
  customer waits. A batching engine has no advantage here and can be slower, since
  continuous batching costs scheduling overhead it cannot amortise over one request.
* Concurrent - N requests in flight. This is throughput, and it is the regime vLLM
  exists for. Quoting a sequential number as "vLLM performance" would understate it;
  quoting only the concurrent number would overstate the experience of one customer.

The comparison backends are the ones this machine can actually run: transformers on
MPS (what Phase 1 generated with), and MLX, which is the native Apple-silicon path
and the fair local competitor for a CPU-only vLLM build.
"""

from __future__ import annotations

import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

from ..llm import ChatClient
from ..logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class BenchResult:
    backend: str
    mode: str
    n_requests: int
    concurrency: int
    wall_clock_s: float
    output_tokens: int
    throughput_tok_s: float
    latency_p50_s: float
    latency_p95_s: float
    latency_mean_s: float
    errors: int = 0
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def _summarise(
    backend: str, mode: str, latencies: list[float], tokens: int, wall: float,
    concurrency: int, errors: int, notes: str = "", extra: dict | None = None,
) -> BenchResult:
    lat = sorted(latencies) or [0.0]
    idx95 = min(len(lat) - 1, int(round(0.95 * (len(lat) - 1))))
    return BenchResult(
        backend=backend,
        mode=mode,
        n_requests=len(latencies),
        concurrency=concurrency,
        wall_clock_s=wall,
        output_tokens=tokens,
        throughput_tok_s=tokens / wall if wall else 0.0,
        latency_p50_s=statistics.median(lat),
        latency_p95_s=lat[idx95],
        latency_mean_s=statistics.mean(lat),
        errors=errors,
        notes=notes,
        extra=extra or {},
    )


def bench_openai_backend(
    client: ChatClient,
    model: str,
    prompts: list[list[dict[str, str]]],
    *,
    backend: str,
    concurrency: int = 1,
    max_tokens: int = 200,
    temperature: float = 0.0,
) -> BenchResult:
    latencies: list[float] = []
    tokens = 0
    errors = 0

    def one(msgs):
        t0 = time.perf_counter()
        try:
            res = client.chat(
                msgs, model=model, temperature=temperature, max_tokens=max_tokens
            )
            return time.perf_counter() - t0, res.completion_tokens, None
        except Exception as exc:  # noqa: BLE001
            return time.perf_counter() - t0, 0, exc

    t0 = time.perf_counter()
    if concurrency <= 1:
        results = [one(m) for m in prompts]
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(pool.map(one, prompts))
    wall = time.perf_counter() - t0

    for lat, tok, err in results:
        if err is not None:
            errors += 1
            continue
        latencies.append(lat)
        tokens += tok

    mode = "sequential" if concurrency <= 1 else f"concurrent-{concurrency}"
    return _summarise(backend, mode, latencies, tokens, wall, concurrency, errors)


def bench_hf(generator, prompts: list[list[dict[str, str]]], *, max_tokens: int = 200) -> BenchResult:
    """transformers on MPS. Batched only - one-at-a-time is not how it is used here."""
    t0 = time.perf_counter()
    generator.generate(
        prompts, temperature=0.0, top_p=1.0, max_tokens=max_tokens, show_progress=False
    )
    wall = time.perf_counter() - t0
    stats = generator.last_stats
    per_request = wall / len(prompts)
    return _summarise(
        "transformers-mps",
        f"batched-{generator.batch_size}",
        [per_request] * len(prompts),
        int(stats.get("output_tokens", 0)),
        wall,
        generator.batch_size,
        0,
        notes=(
            "static batching: every request in a batch waits for the slowest to finish, "
            "so per-request latency here is an average, not a real per-request measurement"
        ),
    )


def bench_mlx(
    model_path: str, prompts: list[list[dict[str, str]]], *, max_tokens: int = 200
) -> BenchResult:
    """MLX on Metal. The native Apple-silicon path, and the fair local comparison."""
    from mlx_lm import generate, load

    model, tokenizer = load(model_path)
    latencies: list[float] = []
    tokens = 0
    t0 = time.perf_counter()
    for msgs in prompts:
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        s = time.perf_counter()
        out = generate(model, tokenizer, prompt=text, max_tokens=max_tokens, verbose=False)
        latencies.append(time.perf_counter() - s)
        tokens += len(tokenizer.encode(out))
    wall = time.perf_counter() - t0
    return _summarise(
        "mlx-metal", "sequential", latencies, tokens, wall, 1, 0,
        notes="mlx_lm has no continuous batching; sequential is the only regime it offers here",
    )
