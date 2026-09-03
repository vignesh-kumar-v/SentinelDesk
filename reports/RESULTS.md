# SentinelDesk — Results

_Generated 2026-09-03 07:58 UTC from the phase reports in this directory. Regenerate with `make report`._

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

_not run_


## Phase 5 — Blind win-rate, held-out

_not run_

