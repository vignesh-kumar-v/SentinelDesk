# SentinelDesk — Results

_Generated 2026-09-04 07:35 UTC from the phase reports in this directory. Regenerate with `make report`._

## Headline

_Phase 5 has not been run. No win-rate to report._


## Phase 0 — DPO loop verified on a toy batch

- loss 0.6931 -> 0.0004, reward margin 0.0 -> 7.8125, accuracy 1.0
- 6 optimiser steps in 9.9s on Qwen/Qwen2.5-0.5B-Instruct
- The starting loss of 0.6931 is ln 2 to four places, which is the independent check that the policy is identical to the reference at init: prompt masking and reference handling are both correct.


## Phase 1 — Preference data

- 594 synthetic tickets, 22 policies covered, median body 50 words
- split: {'heldout': 149, 'train': 445}
- categories: {'billing': 139, 'technical': 135, 'account_access': 106, 'shipping': 107, 'product_info': 107}
- scenario mix: {'multi_part': 109, 'vague': 108, 'wrong_assumption': 211, 'plain': 88, 'urgent_pressure': 78}

## Phase 2 — DPO training

_not run_


## Phase 3 — Agent graph

_not run_


## Phase 4 — Serving

24 requests per arm, max 160 new tokens.

| backend | mode | requests | tokens/s | p50 latency | p95 latency | wall | errors |
|---|---|---|---|---|---|---|---|
| vllm-cpu | sequential | 24 | 17.2 | 9.00s | 10.32s | 205.7s | 0 |
| vllm-cpu | concurrent-8 | 24 | 96.6 | 10.41s | 12.07s | 32.7s | 0 |
| transformers-mps | batched-8 | 24 | 83.2 | 1.18s | 1.18s | 28.4s | 0 |
| mlx-metal | sequential | 8 | 131.7 | 0.42s | 1.56s | 5.0s | 0 |

- _transformers-mps (batched-8): static batching: every request in a batch waits for the slowest to finish, so per-request latency here is an average, not a real per-request measurement_
- _mlx-metal (sequential): mlx_lm has no continuous batching; sequential is the only regime it offers here_

**Continuous batching is worth 5.6x throughput** (17.2 -> 96.6 tokens/s) for 1.16x the median per-request latency. That trade is the reason vLLM exists, and it is invisible in a sequential benchmark — which is why both regimes are measured rather than just the flattering one.

**These backends are not on equal hardware, and the table should not be read as if they were.** vLLM has no Metal backend and ships no macOS wheel, so it runs its CPU build inside a linux/arm64 container; MLX runs on the Metal GPU and transformers on MPS. A GPU path beating a CPU path is the expected outcome, not a verdict on vLLM. What the vLLM rows do establish is the batching behaviour above, which is a property of the engine rather than of the silicon.

## Phase 5 — Blind win-rate, held-out

_not run_

