# SentinelDesk — Project Blueprint

A multi-agent support-ticket system where the resolution agent's quality is *fine-tuned* via preference optimization (DPO), not just prompted — wrapped in industry-standard guardrails and full observability.

---

## 1. Elevator Pitch

Route incoming support tickets through a small agent graph (triage → resolution → escalation). The resolution agent isn't just a prompted LLM call — it's a small model that's been **DPO-tuned** on preference pairs to prefer concise, correct responses over verbose-but-wrong ones, served via vLLM, guarded by an industry-standard guardrails framework, and fully traced end-to-end.

## 2. Why This Project (gaps it's designed to fill)

| Gap on current resume | How this project fills it |
|---|---|
| DPO project (from-scratch) has no applied result yet | Apply it here: real preference pairs, real before/after win-rate |
| No RLHF/alignment story on a resume otherwise full of pretraining/inference work | DPO-tuned agent is the centerpiece, not a side detail |
| vLLM is listed as a skill but no project actually uses it | Serve the DPO-tuned model via vLLM inside the agent graph |
| Guardrail work so far is custom-built only (NeuralScholar) | Pair it with a named industry framework (NeMo Guardrails / Guardrails AI) |
| No LLM observability/tracing across a multi-agent pipeline | Langfuse traces every agent hop: latency, cost, tool calls |
| Cloud section is GCP-only; agent-hosting story is untested | Deploy the agent-hosting layer on AWS Bedrock or Azure, via Terraform |

## 3. Scope: Core vs. Stretch

The DPO fine-tuning is the entire point of this project — it is **not optional** the way stretch items are elsewhere. Everything else exists to serve and demonstrate it credibly.

**Core (must finish, ~3–4 weeks — this project is inherently a notch harder than FraudPulse because of the fine-tuning step):**
1. Preference-pair generation (LLM judge scoring response pairs)
2. DPO fine-tuning of the resolution agent's model (continuation of your existing from-scratch DPO project)
3. LangGraph agent graph (triage → resolution → escalation)
4. vLLM serving of the tuned model inside the graph
5. Evaluation benchmark: DPO-tuned agent vs. base-prompted agent, judge-rated win-rate

**Stretch (only after Core's win-rate result is real and verified):**
6. Named guardrails framework (NeMo Guardrails or Guardrails AI)
7. Langfuse tracing across the full agent pipeline
8. Terraform-provisioned cloud deployment (AWS Bedrock or Azure OpenAI)

Do not start a stretch item until Core Phase 5's benchmark produces an actual, honestly-reported number. A DPO project with no measured win-rate is not a DPO project — it's an unfinished script.

## 4. Architecture Overview

```
[Support ticket input]
        |
        v
  Triage agent (LangGraph node)
  -- classifies ticket type/urgency --
        |
        v
  Resolution agent (LangGraph node)
  -- served via vLLM, running the
     DPO-tuned small model --
        |
        +---- confident? ----> [Response returned]
        |
        v (not confident / policy-flagged)
  Escalation agent (LangGraph node)
  -- routes to human queue --

  [Guardrails layer wraps every agent's output:
   topic rails, PII redaction, structured-output validation]

  [Langfuse traces every hop: input/output, latency,
   token cost, tool calls, judge/faithfulness scores]
```

The point of this diagram: the **resolution agent's model itself is a training artifact**, not a wrapped API call. Everything else in the diagram exists to host, govern, and observe that artifact credibly.

## 5. Data

- **Ticket source:** a public support-ticket dataset (e.g., a Kaggle customer-support-ticket dataset) or synthetically generated tickets across a few categories (billing, technical, account access) — synthetic is fine and faster to control if a good public dataset isn't available.
- **Preference pairs for DPO:** for a sample of tickets, generate two candidate resolution-agent responses per ticket (e.g., one from a stronger/verbose prompting strategy, one from a weaker/rushed one), then have an LLM judge (a frontier model, scored against a rubric: correctness, conciseness, tone) label which is preferred. This becomes the `(prompt, chosen, rejected)` triples DPO needs — the same shape as your existing from-scratch DPO project, just with real, task-relevant data instead of a toy dataset.

## 6. Build Phases — Steps and Verification

### Phase 0 — Setup
- Repo scaffold. Reuse your existing DPO-from-scratch code as the training-loop foundation rather than rewriting it.
- Pick the base model (your existing project already uses Qwen2.5-0.5B-Instruct — a good choice here too, small enough to fine-tune and serve quickly).
- **Verify:** existing DPO training loop runs end-to-end on a tiny toy batch without errors before touching real ticket data.

### Phase 1 — Preference-pair generation
- Generate ~200–500 tickets (synthetic or sampled from a public dataset).
- For each, generate two candidate resolution responses using two different prompting strategies.
- Use an LLM judge with a fixed rubric to label chosen/rejected. Log the rubric itself, not just the labels — you'll need to defend it.
- **Verify:** manually spot-check 20 judge labels yourself. If you disagree with the judge more than ~10–15% of the time, fix the rubric before generating the rest — a DPO run on noisy preference data will produce a noisy result, and you want to know that going in, not discover it after training.

### Phase 2 — DPO fine-tuning
- Run DPO training on the preference pairs using your existing raw-PyTorch implementation.
- Track loss, implicit reward margins, and KL-to-reference-policy over training (the last one especially — DPO runs that drift too far from the reference policy are a common, real failure mode worth watching for and reporting on if it happens).
- **Verify:** training loss decreases and reward margin between chosen/rejected grows; save the checkpoint.

### Phase 3 — LangGraph agent pipeline
- Build the triage → resolution → escalation graph. Triage can be a smaller/cheaper model or simple classifier; resolution is your DPO-tuned model; escalation is a simple rule ("route to human if resolution agent's confidence is below threshold or ticket is flagged").
- **Verify:** run 10 sample tickets through the full graph end-to-end and confirm routing logic behaves as expected (a billing ticket doesn't get misrouted to escalation by default, etc.).

### Phase 4 — vLLM serving
- Serve the DPO-tuned checkpoint via vLLM, and wire the resolution agent node to call it instead of a local Ollama/llama.cpp instance.
- **Verify:** measure tokens/sec and latency via vLLM, and — since you've already benchmarked Ollama/llama.cpp elsewhere — a direct comparison against those is a natural, low-effort addition that strengthens the bullet.

### Phase 5 — Evaluation: the core result
- Build a held-out test set of tickets (not used in preference-pair generation).
- Run both the DPO-tuned agent and a base-prompted-only agent (same base model, no fine-tuning, just a good system prompt) on the test set.
- Use the same LLM-judge rubric from Phase 1 to score win-rate: DPO-tuned vs. base-prompted, blind (judge doesn't know which is which).
- **Verify:** report the actual win-rate, whatever it is. If DPO doesn't clearly beat the base-prompted version, that is itself a legitimate and interesting finding (worth investigating why — reward hacking, insufficient preference data, judge inconsistency) — document it honestly rather than tuning the benchmark until it produces the answer you wanted.

### Phase 6 (stretch) — Named guardrails framework
- Integrate NeMo Guardrails or Guardrails AI for topic rails, PII redaction, and structured-output validation on agent responses.
- **Verify:** feed a few adversarial/off-topic inputs and confirm the guardrail actually blocks or redacts as configured — an untested guardrail is worse than no guardrail bullet at all.

### Phase 7 (stretch) — Langfuse tracing
- Instrument every agent hop (triage, resolution, escalation) with Langfuse: latency, token cost, and — if feasible — the judge/faithfulness score per response.
- **Verify:** pull up a trace for a single ticket end-to-end and confirm every hop is visible with correct timing/cost data, not just the final output.

### Phase 8 (stretch) — Terraform-provisioned cloud deployment
- Provision the agent-hosting layer (e.g., AWS Bedrock endpoint, or the vLLM server on an EC2/ECS instance) via Terraform.
- **Verify:** `terraform apply` / `terraform destroy` cleanly stand up and tear down the deployment.

## 7. Tech Stack Summary

| Layer | Tool | Status |
|---|---|---|
| Preference optimization | DPO (raw PyTorch, extending existing project) | Core |
| Agent orchestration | LangGraph | Core |
| Model serving | vLLM | Core |
| Base model | Qwen2.5-0.5B-Instruct (or similar small model) | Core |
| Judge / eval | Frontier LLM as judge, fixed rubric | Core |
| Guardrails | NeMo Guardrails / Guardrails AI | Stretch |
| Observability | Langfuse | Stretch |
| Infra as code | Terraform (AWS Bedrock or Azure) | Stretch |

## 8. "Done" Definition

The project is resume-worthy at Core scope when all of the following are true, with real numbers behind each claim:

- [ ] Preference pairs generated with a documented, spot-checked judge rubric
- [ ] DPO training completes with visible reward-margin improvement over training
- [ ] Full triage → resolution → escalation graph runs end-to-end on sample tickets
- [ ] Resolution agent is served via vLLM, not a local dev server
- [ ] Blind judge-scored win-rate (DPO-tuned vs. base-prompted) is measured on a held-out set and reported honestly, including if the result is unflattering

## 9. Metrics to Track and Report Honestly

- DPO training: loss curve, chosen/rejected reward margin, KL-to-reference-policy
- Win-rate: DPO-tuned vs. base-prompted, on held-out tickets, judged blind
- vLLM serving: tokens/sec, latency, vs. your existing Ollama/llama.cpp numbers from other projects
- Guardrail true-positive rate on adversarial/off-topic test inputs (if built)
- If the DPO-tuned model doesn't clearly win, or shows a specific failure mode (reward hacking toward short-but-unhelpful responses is a classic one) — that's a finding worth a bullet in itself, the same way the FSDP bug or the TensorRT silent failure were.

## 10. Suggested Timeline (Core scope, ~3–4 weeks)

- **Days 1–2:** Phase 0 (setup, reuse existing DPO code)
- **Days 3–6:** Phase 1 (preference-pair generation + rubric spot-check)
- **Days 7–10:** Phase 2 (DPO fine-tuning)
- **Days 11–14:** Phase 3 (LangGraph pipeline) + Phase 4 (vLLM serving)
- **Days 15–18:** Phase 5 (evaluation benchmark) + write-up / resume bullet draft
- **If ahead of schedule:** pick *one* stretch phase — Langfuse tracing is likely the highest-leverage single addition if you can only fit one

## 11. Target Resume Bullet (draft — fill in real numbers once built)

> Fine-tuned a support-resolution agent via DPO on judge-labeled preference pairs (raw PyTorch), improving blind judge-rated win-rate by [X]% over a base-prompted baseline; served the tuned model via vLLM within a LangGraph triage/resolution/escalation pipeline[, guarded by NeMo Guardrails and traced end-to-end via Langfuse].

Only include the guardrails/Langfuse clause if those stretch phases were actually built and verified — same rule as FraudPulse.

## 12. Risks / Open Questions

- LLM-as-judge introduces its own noise and bias; the Phase 1 spot-check step exists specifically to catch this early rather than discover it after a full DPO run.
- DPO on a very small model (0.5B) may show a smaller or noisier effect than on a larger model — worth setting expectations accordingly, and worth explicitly noting the model size as a limitation if the result is modest.
- vLLM's setup overhead (GPU memory requirements, batching config) is nontrivial for a first-time user — budget real time for this in Phase 4 rather than treating it as a quick swap-in.
- Decide early whether the escalation agent's "confidence threshold" is rule-based (simpler, faster to build) or itself model-based (more sophisticated, more time) — rule-based is the right choice given the DPO work is where the real depth should go.
