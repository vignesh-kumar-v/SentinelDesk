# SentinelDesk — Results

_Generated 2026-09-04 10:35 UTC from the phase reports in this directory. Regenerate with `make report`._

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
- judge: `deepseek-v4-pro:cloud`, rubric `v1` (fingerprint `f14d5cd4800b`)
- 409 comparisons x 2 display orders
- **order-inconsistency rate 15.9%** — how often swapping which response was shown first flipped the verdict. Those comparisons are dropped rather than resolved, and this rate bounds how much any downstream win-rate can be trusted.
- 344 usable pairs (84.1% yield)
- chosen-strategy split: {'rushed': 123, 'grounded': 221}
- mean rubric scores by strategy: {"grounded": {"correctness": 0.489, "completeness": 0.292, "conciseness": 1.127, "tone": 0.961, "total": 2.869}, "rushed": {"correctness": 0.417, "completeness": 0.049, "conciseness": 0.197, "tone": 0.639, "total": 1.302}}
- length ratio chosen/rejected: 0.662 (a ratio far from 1.0 means DPO would learn length before it learns correctness)

**Which rubric dimension actually decided the training labels** (over the 344 usable pairs). If tone or conciseness separated them and correctness did not, DPO would be learning house style and any win-rate would be measuring that instead.

| dimension | mean winner−loser gap | favours winner in |
|---|---|---|
| correctness | +0.738 | 68.9% of pairs |
| completeness | +0.157 | 26.2% of pairs |
| conciseness | +0.605 | 67.7% of pairs |
| tone | +0.449 | 61.3% of pairs |

- correctness was identical on both sides in 31.1% of pairs; both sides scored 0 on correctness in 25.9%
- **second-judge cross-check**: `kimi-k3:cloud` re-labelled 40 of the same comparisons — raw agreement 72.5%, Cohen's kappa 0.5857
- **agreement where both judges committed to a winner: 24/25 = 96.0%**. This is the number that bears on the preference labels. Two labellers can disagree about which response is better, or about whether the gap is decisive enough to call; only the first threatens the data, and a "tie" here means the primary judge flipped across display orders, which already excludes that comparison from training.
- confusion (primary|second): `{'grounded|grounded': 12, 'rushed|grounded': 1, 'tie|tie': 5, 'rushed|rushed': 12, 'tie|grounded': 5, 'grounded|tie': 2, 'tie|rushed': 3}`

## Phase 2 — DPO training

_not run_


## Phase 3 — Agent graph

- 10 tickets end-to-end through triage -> retrieve -> resolution -> gate
- triage accuracy vs the ticket's true category: **100.0%**
- escalation rate 20.0% (2/10)
- resolver backend: remote

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

