# SentinelDesk

A support-ticket agent pipeline whose resolution model is a **training artifact**, not a
prompted API call. A small model is fine-tuned with **DPO** — implemented from scratch in
raw PyTorch — on preference pairs labelled by a frontier LLM judge against a fixed rubric,
then served inside a LangGraph triage → resolution → escalation graph and measured against
its own un-tuned base in a blind, both-orders judge arena on held-out tickets.

The headline number, the training curves and every supporting metric live in
**[reports/RESULTS.md](reports/RESULTS.md)**, generated from the run artifacts so the
write-up cannot drift from the runs. Design choices and the things that broke are in
**[docs/decisions.md](docs/decisions.md)**.

---

## Why the setup looks like this

Most "LLM agent" projects are prompts around an API. The question here is whether
preference optimization on a 0.5B model produces a measurably better support agent than
the same model with a good prompt — and whether that difference survives an evaluation
designed to *not* flatter it.

Three choices carry most of the project's weight:

**Correctness is checkable, not a judge's opinion.** `configs/policies.yaml` holds 22
ground-truth support policies. Every ticket is generated *from* one; the judge grades both
candidate responses against it. Without this an LLM judge can only score tone and fluency,
and DPO trained on tone-only preferences teaches confident writing rather than accuracy —
exactly the failure mode this project needs to be able to detect. About a third of tickets
carry a confidently wrong customer assumption the policy contradicts, because agreeing with
the customer is the path of least resistance for a weak model.

**Both candidates come from the model being tuned.** Generating the "chosen" response with
a frontier model would make this distillation dressed as preference learning, and DPO's
implicit reward would be defined over responses the policy could never produce. Both come
from Qwen2.5-0.5B-Instruct under two different system prompts and sampling temperatures.

**Every comparison is judged twice, with the order swapped.** LLM judges favour whichever
response they see first. Judging once folds that bias into the training data where it is
indistinguishable from signal. A label survives only if the judge agrees with itself across
the swap, and the disagreement rate becomes a reported number that bounds how much any
downstream win-rate can be trusted.

---

## Architecture

```
                       ticket
                         │
                         ▼
   ┌───────────────────────────────────────────┐
   │ triage      LLM classifier, keyword       │  category, urgency,
   │             fallback if the model is down │  triage confidence
   └───────────────────────────────────────────┘
                         │
                         ▼
   ┌───────────────────────────────────────────┐
   │ retrieve    policy excerpts for the       │  the agent gets the whole
   │             triaged category              │  category, not the one answer
   └───────────────────────────────────────────┘
                         │
                         ▼
   ┌───────────────────────────────────────────┐
   │ resolution  ◄── the DPO-tuned checkpoint, │  draft, latency, tokens
   │                 served over OpenAI proto  │
   └───────────────────────────────────────────┘
                         │
                         ▼
   ┌───────────────────────────────────────────┐
   │ gate        rule-based: groundedness,     │  confidence + named reasons
   │             hedging, repetition, length,  │
   │             mandatory-escalation policies │
   └───────────────────────────────────────────┘
                    │           │
        confident   │           │  low confidence, empty draft,
                    ▼           ▼  or a policy that forbids automation
              ┌─────────┐  ┌──────────────┐
              │ respond │  │ escalate     │ → human-tier-2
              └─────────┘  └──────────────┘
```

The gate is rule-based deliberately. The *reasons* are the product: "escalated because the
draft contradicted ACC-02" is auditable in a way a learned head emitting `0.41` is not, and
the depth in this project belongs in the DPO work. Some requests escalate regardless of
score — 2FA disable, GDPR erasure, SLA credits, invoice entity changes — because on those a
fluent, confident draft is the dangerous outcome, not a reassuring one.

---

## The DPO implementation

`src/sentineldesk/dpo/` is written out rather than imported from TRL, because the
reference-policy behaviour is the part worth being able to inspect.

```
r_θ(x, y) = β · ( log π_θ(y|x) − log π_ref(y|x) )
L_DPO     = −log σ( r_θ(x, y_w) − r_θ(x, y_l) )
```

What the training loop logs, and why:

| Metric | Why it is tracked separately |
|---|---|
| loss | Starts at `ln 2 = 0.6931` when the policy equals the reference. A start anywhere else means prompt masking or reference handling is broken — the cheapest correctness check available. |
| chosen / rejected implicit reward | Logged **apart**, not just as their difference. A run where both march downward while the margin grows is a policy collapsing away from the reference, and the margin curve alone reports that as success. |
| reward margin | What the loss actually optimises. |
| preference accuracy | Fraction of pairs where the margin is positive. |
| `log π − log π_ref` | Drift from the reference policy, the failure mode the blueprint calls out. |

Two implementation notes. The reference model's log-probs are computed once up front and
the model is then freed — it is frozen and never updated, so keeping it resident costs a
second forward pass every step and half of peak memory for nothing. And prompts are
tokenised through the model's own chat template with the same system prompt used at serving
time, from a single shared `prompts.py`; a mismatch there is invisible in the loss curve and
surfaces only as a tuned model that underperforms its own training metrics.

---

## Evaluation

The Phase 5 arena is built to make the number hard to inflate:

- Both arms get the **identical system prompt**, the **identical retrieved policies** and the
  **identical decoding settings**. The only difference is the weights. A baseline handicapped
  by a worse prompt turns a prompt-engineering win into a claimed fine-tuning win.
- **Held-out tickets only** — split off before any preference pair was built.
- Each comparison is judged in **both display orders**; order-flips are reported as ties
  rather than silently resolved. Arm-to-slot assignment is randomised on top of that.
- The **same rubric as Phase 1**, unrevised. Scoring the benchmark with a changed rubric
  measures the rubric change, not the fine-tune.
- **Two win-rates** are reported — ties dropped, and ties counted as half. Quoting only the
  flattering one is the standard way this metric gets abused, and a fine-tune that produces
  more ties has not improved anything.
- Wilson score intervals and an **exact** binomial test, not the normal approximation, which
  at these sample sizes produces intervals running outside `[0, 1]`.
- Wins are correlated against **response-length delta** — the reward-hacking check. A
  strongly negative correlation would mean the tuned arm wins by being terse, not by being
  right.

Judge trustworthiness is measured, not assumed: an independent second frontier model
re-labels a stratified sample of the same comparisons, reported as raw agreement **and**
Cohen's κ. κ rather than raw agreement because these labels are skewed — a labeller that
always picked the majority class would score high raw agreement while carrying no
information. `make spotcheck-human` writes a blind worksheet for a human pass as well.

---

## Layout

```
configs/policies.yaml          22 ground-truth support policies
src/sentineldesk/
  prompts.py                   the resolution prompt, defined once
  llm.py                       one OpenAI-compatible client for every model
  data/       generate.py      synthetic tickets, grounded + stratified
              kb.py            policy retrieval
  prefs/      rubric.py        the judge rubric, versioned + fingerprinted
              judge.py         both-orders judging with position-bias control
              candidates.py    the two on-policy candidate strategies
              build_pairs.py   resumable candidate generation + judging
              spotcheck.py     second-judge and human agreement, Cohen's κ
  dpo/        loss.py          the DPO objective, from scratch
              train.py         raw PyTorch loop with cached reference log-probs
              dataset.py       chat-template tokenisation + prompt masking
  agents/     graph.py         the LangGraph pipeline
              confidence.py    rule-based gate with named reasons
  serving/    vllm_server.py   vLLM process management (docker + native backends)
              local.py         batched transformers baseline
              bench.py         throughput/latency, sequential and concurrent
  eval/       arena.py         the blind head-to-head
              stats.py         Wilson intervals, exact binomial, Pearson
  guardrails/ rails.py         NeMo topic rails + deterministic checks
              pii.py           regex PII detection, Luhn-checked cards
              schema.py        response validation, invented-policy-id check
              evaluate.py      TPR/FPR against labelled adversarial cases
  observability/tracing.py     Langfuse, one typed observation per graph hop
configs/guardrails/            rail config + the labelled adversarial set
docker/                        vLLM image pins, self-hosted Langfuse stack
terraform/                     vLLM on ECS Fargate behind an ALB
scripts/build_vllm_macos.sh    vLLM CPU build for Apple silicon
scripts/build_vllm_docker.sh   vLLM CPU image for linux/arm64 (the one that serves)
```

---

## Running it

```bash
make setup      # install into the dev venv
make doctor     # check torch, device and judge reachability
```

Then one target per phase gate:

```bash
make smoke       # PHASE 0  DPO loop end-to-end on a toy batch
make tickets     # PHASE 1a synthetic tickets from the policy KB
make pairs       # PHASE 1b two candidates each, judged in both orders
make spotcheck   # PHASE 1  gate: second-judge agreement + Cohen's κ
make train       # PHASE 2  DPO fine-tune
make curves      # PHASE 2  loss / rewards / margin / drift plots
make graph-check # PHASE 3  gate: routing behaviour on sample tickets
make bench       # PHASE 4  tokens/s and latency across backends
make arena       # PHASE 5  blind win-rate on held-out tickets
make report      # regenerate reports/RESULTS.md
```

Stretch phases:

```bash
make guardrails   # PHASE 6 gate: score the rails on labelled adversarial + benign cases
make langfuse-up  # start self-hosted Langfuse, then
make trace        # PHASE 7 gate: traced tickets, read back off the server
make tf-plan      # PHASE 8: validate + plan (free)
make tf-apply     # BILLABLE — see terraform/README.md
make tf-destroy
```

`make test` runs the suite; `make lint` runs ruff.

`make pairs` is resumable. Candidate generation and judging both checkpoint to
`data/prefs/` as they complete and skip what is already there on a re-run — the judging
stage is ~900 hosted-model calls and an all-or-nothing job of that length loses everything
to a laptop sleeping.

### The judge

Any OpenAI-compatible endpoint works. The default routes through Ollama, which can serve
hosted frontier models via `:cloud` tags without a separate provider key:

```bash
SD_JUDGE_BASE_URL=http://localhost:11434/v1
SD_JUDGE_MODEL=deepseek-v4-pro:cloud
```

See `.env.example` for the full set.

### vLLM on Apple silicon

vLLM ships no macOS wheel and has no Metal backend, so serving means the CPU backend.
`scripts/build_vllm_macos.sh` compiles it from source into its own virtualenv — the serving
environment pins `torch 2.8.0` CPU and the training environment must not be dragged to it.
The script handles three failures that do not announce themselves; they and the current
runtime status are documented in [docs/decisions.md](docs/decisions.md) under **F1**.

---

## Guardrails, tracing and infrastructure

**Guardrails (Phase 6)** are split by what kind of question each check answers. PII is
regex — auditable, microseconds, cannot be prompt-injected, and card numbers are
Luhn-checked so a 16-digit order reference survives. Topic and jailbreak rails are NeMo
Guardrails, because "is this an instruction override" has fuzzy edges. Schema is
pydantic, including a check for cited policy ids that do not exist — a fabricated
"[BIL-09]" reads as authoritative and is unfalsifiable to a customer. The rail model is
deliberately not the resolution model.

Scored against a labelled set rather than demonstrated: **95% true-positive rate at 0%
false-positive rate**. The FPR is the number worth defending — the benign half is built
to be hostile to an over-eager rail (furious customers, cancellation threats, requests
support must decline), because a rail that blocks everything scores a perfect TPR and
takes a support queue offline.

**Tracing (Phase 7)** gives each hop a typed Langfuse observation rather than a generic
span: retrieval is a `retriever`, the rails are `guardrail`s, resolution is a
`generation` carrying model and token counts. Verified by fetching traces back through
the public API — the first run reported success from the SDK while storing nothing,
because Langfuse uploads asynchronously and the failure never reached the client.
Tracing degrades to a no-op when Langfuse is absent and records why.

**Infrastructure (Phase 8)** is the same arm64 vLLM image on ECS Fargate behind an ALB.
`terraform apply` and `terraform destroy` were both run against a real account — 15
resources up, 15 down, confirmed through the AWS API rather than Terraform's exit code.
See `terraform/README.md` for the cost and security trade-offs it makes explicit.

## Limitations

- **The base model is 0.5B.** DPO's effect on a model this small is smaller and noisier than
  it would be at 7B. Where the result is modest, model size is a live explanation and is
  reported as one rather than explained away.
- **Tickets are synthetic.** They are grounded in a hand-written policy set, which makes
  correctness checkable but also makes the distribution cleaner than a real support queue.
- **The judge is an LLM.** Its order-sensitivity is measured and reported, and cross-checked
  against a second frontier model, but it remains a proxy for human preference.
