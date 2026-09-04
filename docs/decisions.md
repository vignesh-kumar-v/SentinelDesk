# Decisions and findings

Running notes on choices that were not obvious, and on things that broke. Written
as they happened rather than reconstructed afterwards.

## D1 — Tickets are generated *from* a ground-truth policy

The judge needs something to grade correctness against. Without a known-correct
answer an LLM judge can only score tone, fluency and length — and DPO trained on
tone-only preferences teaches confident writing, not accuracy, which is precisely
the failure this project is supposed to be able to detect.

So `configs/policies.yaml` holds 22 hand-written support policies, each ticket is
generated from one, and the judge sees the policy excerpts alongside both
candidate responses. Roughly a third of tickets carry a confidently wrong customer
assumption the policy contradicts; those are where a weak model separates from a
good one, because agreeing with the customer is the path of least resistance.

The resolution agent is given the whole *category's* policies, not the one policy
its ticket came from. Handing it exactly the right excerpt would make the task
trivial and flatten the preference signal.

## D2 — Both DPO candidates come from the base model, not a teacher

The obvious way to get a strong "chosen" response is to generate it with a frontier
model. That would make DPO into distillation wearing preference-learning's clothes,
and the implicit reward would be defined over responses the policy could never have
produced. Both candidates come from Qwen2.5-0.5B-Instruct; the two strategies differ
only in system prompt and sampling temperature.

The pair records the *canonical* prompt for both sides, since DPO conditions chosen
and rejected on a single shared prompt. This mirrors how UltraFeedback-style datasets
are built.

## D3 — Every comparison is judged in both display orders

LLM judges have a documented bias toward whichever response appears first. Judging
once folds that bias into the training data where it is indistinguishable from
signal. Judging twice with the order swapped costs 2x the calls and buys two things:
a label only survives if the judge agrees with itself across the swap, and the
disagreement rate becomes a measured number that bounds how much any downstream
win-rate can be trusted.

## D4 — Correctness overrides the rubric total, rather than being weighted heavily

An early version scored four dimensions and took the higher total. On inspection the
judge kept preferring fluent wrong answers over clumsy right ones, because three
writing dimensions outvote one correctness dimension. The rubric now states: if
exactly one response scores 0 on correctness, the other wins outright. Length and
formatting are named as explicit non-criteria, since one of the two candidate
strategies is verbose by construction and an unaddressed verbosity bias would make
the labels mostly a measurement of which prompt was wordier.

## D5 — Reference log-probs are computed once and the reference model is freed

The DPO reference policy is frozen and never updated, so re-running it every step is
a second forward pass per batch and a second model resident in memory, both for
nothing. One pass up front caches them; the reference model is then released. On a
24 GB unified-memory machine this is the difference between comfortable and tight.
The `cache_reference` flag exists so the naive path is still reachable if the
reference is ever made non-static.

## D6 — The escalation gate is rule-based

Per the blueprint, and for a second reason: the reasons are the product. "Escalated
because the draft contradicted ACC-02" is something a support org can audit; a
learned confidence head emitting 0.41 is not. Some requests escalate regardless of
the score — 2FA disable, GDPR erasure, SLA credits, invoice entity changes — because
on those a fluent, confident draft is the dangerous outcome, not a reassuring one.

## D7 — Wilson intervals and an exact binomial test, not the normal approximation

With ~150 held-out tickets and a win-rate that may sit near a boundary, the normal
approximation produces intervals that run outside [0, 1] and a p-value that is not
trustworthy. Both are written out in `eval/stats.py` and checked in tests against
published values.

## F1 — vLLM on Apple silicon: builds, then deadlocks at device init

vLLM ships no macOS wheel (PyPI has Linux only) and has no Metal/MPS backend, so
serving on this machine means compiling the CPU backend from source. That part works
— `vllm 0.11.0+cpu` builds and installs — after three fixes, none of which announce
themselves:

1. `--no-build-isolation` runs the build backend in the target environment, so
   vLLM's own build dependencies have to be installed first. Without them the
   metadata step dies on `ModuleNotFoundError: setuptools_scm`, which reads like a
   vLLM packaging bug and is not one.
2. `requirements/cpu.txt` pins `torch==2.8.0` for Darwin but leaves `torchaudio`
   unpinned, so pip resolves the newest torchaudio — built against a different
   torch. The install succeeds. The failure surfaces much later, inside the running
   server, as `dlopen ... Symbol not found: _torch_library_impl`.
3. vLLM 0.11 requires `transformers >= 4.55.2` with no upper bound. transformers 5
   removed `all_special_tokens_extended`, which vLLM's tokenizer cache calls, so a
   fresh install resolves to a transformers that vLLM cannot use.

What does not work is startup. The CPU worker reaches
`cpu_worker.py:66 init_cpu_threads_env` and then sits at 0% CPU indefinitely. This
reproduces with the API server and with the offline `LLM` class, with the V1 engine
in-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) and in its spawned subprocess, and
with `--enforce-eager` (ruling out a slow inductor compile — a compile would burn
CPU, and this burns none).

A `faulthandler.dump_traceback_later(90, exit=True)` watchdog armed before the call
also never fired, which is the informative part: that watchdog runs on its own C
thread and does not need the GIL. A hang that takes the watchdog down with it points
below Python, at the OpenMP runtime — libomp is not fork-safe, and macOS is where
that bites.

### A false lead worth recording

An earlier run failed differently, with
`AttributeError: module 'ray' has no attribute 'is_initialized'`, and the tidy
explanation was that vLLM's editable install leaks its subpackages into the
top-level namespace so `import ray` resolves to `vllm/ray`. That was wrong. The test
script had been written to `/tmp`, a stray `/tmp/ray` directory left by another
project was on `sys.path[0]`, and it shadowed the real module as a namespace
package. Running the identical script from a different directory resolves `ray`
correctly. The lesson is narrow and practical: scratch scripts do not belong in a
shared temp directory.

## F2 — vLLM does run here, in a linux/arm64 container

The macOS-native deadlock in F1 was never resolved; it was routed around. The same
vLLM CPU backend, built for `linux/arm64` and run under Docker, starts and serves
normally — which is itself evidence for the F1 diagnosis, since the only thing that
changed is the OS runtime underneath the same code.

Two more failures on the way there, both worth recording because neither announces
itself as what it is:

**The container build died with `cannot allocate memory`.** vLLM's `Dockerfile.cpu`
compiles with every available core. Docker Desktop here exposes 15 CPUs against
about 8 GB, and fifteen parallel compiles of vLLM's kernels exhaust it. `setup.py`
honours `MAX_JOBS`, but the upstream Dockerfile declares no `ARG` for it, so
`scripts/build_vllm_docker.sh` patches a copy to add one and leaves the vendored
source alone. The patch asserts the stage it targets still exists, so a vLLM version
bump fails loudly rather than quietly producing an unpatched build.

**The image reproduced both dependency failures from F1 exactly.** torchaudio 2.11
against torch 2.8, and transformers 5.16 — the same two open version ranges, resolved
the same wrong way, killing the server at import rather than at install. They are
pinned in `docker/Dockerfile.vllm-pins`, a thin layer on top of the built image so
that adjusting a pin does not invalidate the kernel compile.

The practical lesson is about the shape of these bugs rather than any one of them:
every failure in F1 and F2 surfaced far from its cause. An unpinned transitive
dependency became a `dlopen` symbol error inside a running server. A missing build
dependency became what looked like a vLLM packaging bug. A memory limit became what
looked like a compiler failure. The version constraint is the thing to read first.

## F3 — Interim read on the preference data, recorded before the arena runs

Written at 83 of 445 comparisons judged, deliberately before any Phase 5 number
exists, so that whatever the arena says later cannot be dressed up as having been
expected. Numbers below are from that partial sample; the final figures over all 409
judged comparisons follow in F4, and one of them moved a lot.

**Order-inconsistency is 22.9%.** Nearly a quarter of comparisons flip when the two
responses swap display position. Those are dropped rather than resolved, so they do
not poison training — but the rate is the honest ceiling on this judge. A Phase 5
win-rate inside a few points of 50% would not be distinguishable from this noise, and
should not be reported as if it were.

**A first read of the aggregate scores was misleading, and the correction matters.**
Averaged across every comparison, the two candidate strategies look nearly identical
on correctness (0.44 vs 0.48 out of 3) while differing a lot on total (2.72 vs 1.39).
Read that way it says the preference is driven by style, and DPO is about to learn
tone — the exact failure the rubric's correctness override was written to prevent.

Per-pair, on the pairs that actually become training data, it is the opposite:

| dimension | mean winner−loser gap | favours winner in |
|---|---|---|
| correctness | +0.797 | 75% of pairs |
| conciseness | +0.547 | 59% |
| tone | +0.320 | 52% |
| completeness | +0.102 | 22% |

Correctness is the largest separator and decides three pairs in four. The aggregate
means hid it because on-policy sampling makes the two strategies similar *on average*
while differing per instance — which is what on-policy preference data is supposed to
look like. Comparing strategy means answers a much weaker question than comparing
sides within a pair, and it is the within-pair comparison that describes the gradient.

This analysis is now computed in `summarise()` and printed in the results report,
rather than living in a one-off script, because it is the check on whether the rubric
did its job.

**What is genuinely weak.** Both candidates score 0 on correctness in 20% of usable
pairs, and correctness is identical on both sides in 25%. In that quarter of the data
the gradient really is coming from style alone. That is a property of a 0.5B model
answering policy questions, not of the rubric, and it caps how much correctness signal
DPO can extract here however good the labelling is.

## F4 — Phase 1 final, and a length confound worth naming

Over all 409 judged comparisons rather than the first 83:

* **Order-inconsistency is 15.9%**, not the 22.9% the partial sample suggested. The
  early estimate was pessimistic; the correction is in the honest direction but it is
  a correction either way, and the partial figure should not be quoted.
* **344 usable pairs** from 445 tickets, an 84% yield.
* `grounded` is chosen 221 times to `rushed`'s 123. The label is not predictable from
  the strategy, which is what on-policy preference data should look like — if one
  prompting strategy won every time, DPO would be learning to imitate that prompt.

Which dimension actually decided the labels, per pair:

| dimension | mean winner−loser gap | favours winner in |
|---|---|---|
| correctness | +0.738 | 68.9% |
| conciseness | +0.605 | 67.7% |
| tone | +0.449 | 61.3% |
| completeness | +0.157 | 26.2% |

Correctness leads, but conciseness is close behind, and that matters because of the
next number.

**The confound: chosen responses are 34% shorter than rejected ones** (606 vs 916
characters, ratio 0.66). The `rushed` strategy is verbose by construction — 971
characters against `grounded`'s 547 — and it loses about two thirds of the time. So
"shorter" and "preferred" are correlated in the training data.

The rubric names length as an explicit non-criterion, and that governs how the judge
*scores*. It cannot decorrelate what the two strategies actually produce. A model
trained on these pairs can lower the loss either by being more correct or by being
shorter, and nothing in the objective distinguishes those.

This is why `corr_win_vs_length_delta` exists in the Phase 5 arena, and it is now
load-bearing rather than decorative. If the tuned arm wins mainly where it is shorter
than the baseline, that correlation will show it, and the honest reading of a win
would be "DPO learned brevity", not "DPO learned accuracy".

Also recorded: in 31% of usable pairs the two sides score identically on correctness
and in 26% both score zero. In roughly a third of the data the gradient carries no
correctness signal at all. That is a property of a 0.5B model answering policy
questions, and it caps what this experiment can show however clean the labelling is.

## F5 — Why the training loop was slow, and why the first diagnosis was wrong

Training initially ran at 5.88 s/pair, which put a three-config sweep at roughly six
hours. Three settings took it to 0.70 s/pair, and all three change peak memory per
step rather than work per step:

| batch | chunk | response cap | s/pair |
|---|---|---|---|
| 2 | 256 | 448 | 5.88 |
| 2 | 64 | 256 | 1.17 |
| 1 | 64 | 256 | 0.70 |

Batch 1 with `grad_accum` 16 beats batch 2 with `grad_accum` 8 *per pair*, which is
backwards if you count arithmetic and correct if you count allocations: on unified
memory a step that fits stays on the GPU and a step that does not thrashes swap.

The response cap was **not** taken from that table. Measured against the data,
responses reach 263 tokens, so a 256-token cap truncates 16% of them — and truncating
a response changes its sequence log-prob and therefore the gradient. That is a silent
bias, not a speed/quality trade, so the cap is 320, which truncates nothing and keeps
the speedup.

**The wrong diagnosis.** Before this, left padding and bfloat16 were reported as having
made training *slower* — 16 s/it, then 39-48 s/it. That was wrong. An earlier
micro-benchmark process was still alive and holding MPS memory, and swap had filled to
33 GB of 34 GB; every timing taken in that window was measuring contention. The lesson
is procedural rather than technical: on a machine where the accelerator shares memory
with everything else, a timing number is only meaningful with a verified-quiet machine,
and "did my last experiment actually exit" is part of the measurement.

## F6 — The Phase 1 judge gate, and why raw agreement is the wrong headline

An independent frontier model (`kimi-k3:cloud`) re-labelled 40 of the same
comparisons under the identical rubric, in both display orders:

| metric | value |
|---|---|
| raw agreement | 72.5% |
| Cohen's kappa | 0.586 |
| agreement where both judges committed to a winner | 24/25 = **96.0%** |

The confusion matrix is where the interpretation lives:

```
grounded|grounded 12   rushed|rushed 12   tie|tie 5
tie|grounded       5   tie|rushed     3   grounded|tie 2
rushed|grounded    1
```

Of eleven disagreements, **one** is a direct winner flip. The other ten have a "tie"
on one side, and a tie from the primary judge means it flipped across display orders —
so that comparison was already dropped before training. The two judges essentially
never disagree about which response is better; they disagree about whether the gap is
decisive enough to call.

Those are different failures and only the first threatens the labels, so
`decisive_agreement` is computed and reported next to kappa rather than being folded
into it. Reporting 72.5% alone would understate a judge that is directionally
reliable; reporting 96% alone would hide how often it declines to commit. Both are in
the results.

Against the blueprint's own gate — "fix the rubric if you disagree more than about
10-15%" — raw disagreement is 27.5% and fails it, directional disagreement is 4% and
passes. The rubric is kept, with that reasoning stated rather than the flattering half
quoted.

Kappa is reported rather than raw agreement alone for the usual reason: these labels
are skewed, and a labeller that always picked `grounded` would score high raw
agreement while carrying no information. 0.586 is "moderate" on the Landis-Koch scale,
just short of "substantial".

## F7 — Both implicit rewards fall; the margin grows because the rejected side falls faster

Recorded during the sweep, before the arena runs.

The `lr=2e-6` run looks healthy on every headline metric: training loss 0.715 → 0.486
at its best, eval margin −0.006 → +0.155, eval preference accuracy 0.47 → 0.62. A
margin curve alone would call this a success.

Splitting the margin into its two halves says something different:

| step | loss | r_chosen | r_rejected | margin | acc |
|---|---|---|---|---|---|
| 0 | 0.7153 | **+0.027** | +0.067 | −0.041 | 0.38 |
| 8 | 0.6836 | **−0.033** | −0.057 | +0.024 | 0.62 |
| 16 | 0.6962 | **−0.240** | −0.275 | +0.035 | 0.50 |
| 24 | 0.5831 | **−0.224** | −0.544 | +0.320 | 0.62 |
| 32 | 0.4857 | **−0.116** | −0.644 | +0.528 | 0.81 |
| 40 | 0.6321 | **−0.275** | −0.515 | +0.240 | 0.56 |

`r_chosen` is the implicit reward on the *preferred* response, and it goes **down**,
not up. The model becomes less likely to produce the chosen responses in absolute
terms. The margin grows only because `r_rejected` falls faster.

This is the failure mode the loss module was written to expose — the docstring there
says a run where both rewards march downward while the margin grows is a policy
moving away from the reference, and the margin curve reports that as success. It is a
documented property of DPO rather than a bug in this implementation: the objective
constrains only the *difference* of the two log-ratios, so pushing both down satisfies
it as readily as pulling the chosen one up.

**What it predicts for Phase 5.** A model that has become less likely to emit the
preferred responses may well generate worse text overall while still ranking chosen
above rejected more reliably. Preference accuracy on held-out *pairs* and win-rate on
held-out *tickets* are different quantities, and this is precisely the regime where
they can point in opposite directions. Whatever the arena returns, that is the reading
to check first — which is why this is written down before the arena has run.

Drift stays modest in absolute terms (`kl/chosen` around −2.7 against a limit of 30),
so this is not the runaway-divergence case. It is the quieter version: the policy is
still close to the reference, but it is moving in a direction that lowers the
probability of good and bad responses alike.

## F8 — The A/A null test, and the noise floor it exposes

Before trusting any win-rate, the harness was run with the *same model in both arms* —
two samples from identical weights at temperature 0.7, differing only by seed. The
true win-rate is 50% by construction, so anything the harness reports beyond noise is
the apparatus leaking a signal.

| metric | value |
|---|---|
| wins / losses / ties | 15 / 8 / 7 |
| win-rate, tie-adjusted | 0.617 |
| 95% CI (Wilson) | [0.439, 0.768] |
| binomial p vs 50% | 0.21 |
| order-inconsistency | 23.3% |

**It passes**: 50% sits inside the interval and the deviation is not significant. There
is no detectable position bias surviving the both-orders swap, and no arm-to-slot
correlation.

**But the point estimate is 0.617, from identical weights.** That is the number to
keep in mind when reading the real arena. On 30 tickets, sampling noise alone can
produce what looks like a 12-point advantage. A DPO win-rate in that range, on a
sample that size, would be indistinguishable from nothing happening. The real arena
runs on 149 tickets, which narrows the interval considerably, but the shape of the
lesson survives: the confidence interval is not decoration here, it is the finding.

A secondary observation worth recording: order-inconsistency is **higher** in the A/A
test (23.3%) than in Phase 1 labelling (15.9%). That is the right direction. Two
samples from one model are genuinely near-equivalent, so the judge has less to grip
and flips more often when the order changes. A judge whose inconsistency stayed flat
between "these differ" and "these are identical" would be one whose verdicts were not
tracking response quality at all.

## F9 — The Phase 5 result: DPO learned brevity

**52W / 32L / 65T on 149 held-out tickets, judged blind in both display orders.**

| framing | win-rate | 95% CI | 50% inside? |
|---|---|---|---|
| ties counted as half | 56.7% | 48.7–64.4% | **yes — not significant** |
| ties dropped | 61.9% | 51.2–71.5% | no — significant (p = 0.038) |

The two framings disagree, and which one is quoted decides whether this clears
significance. With 65 of 149 comparisons tied, the conservative reading is the honest
one: a fine-tune that mostly produces ties has not improved much. **The headline is
56.7%, an interval that contains 50%.**

### What actually improved

| dimension | tuned | base | gap | share of total gap |
|---|---|---|---|---|
| conciseness | 1.46 | 1.11 | **+0.349** | **81.7%** |
| correctness | 0.53 | 0.47 | +0.054 | 12.6% |
| completeness | 0.35 | 0.32 | +0.031 | 7.3% |
| tone | 1.14 | 1.15 | −0.007 | −1.6% |

Four fifths of the advantage is conciseness. Correctness — the dimension the rubric
was built to make decisive, and the one that matters for a support agent — moved by
0.054 on a 0-3 scale.

### The three pre-registered predictions all held

Each was written down before this run, in F4, F7 and F8.

**F4, the length confound.** The preference pairs had the chosen side at 0.662x the
length of the rejected side. The tuned model now writes at **0.683x** the baseline's
length. Those ratios matching to within two points is the clearest single piece of
evidence that what DPO extracted from this data was substantially *length*. Wins also
correlate with being shorter (r = −0.278).

**F7, the falling implicit rewards.** The prediction was that preference accuracy on
held-out *pairs* and win-rate on held-out *tickets* could diverge, because `r_chosen`
was decreasing. Pair accuracy rose 0.47 → 0.62, a 15-point gain. Ticket win-rate is
56.7% with 50% inside the interval. They diverged.

**F8, the noise floor.** The A/A null test returned 0.617 from two runs of *identical*
weights. The real result, 56.7%, sits **below** that point estimate. The A/A ran on 30
tickets and the arena on 149, so the intervals differ — but it is a useful reminder of
how little a mid-50s win-rate is worth on samples this size.

### Why order-inconsistency is 43.6% here, against 15.9% in Phase 1

Not a defect. The judge flips exactly where there is nothing to choose between:

| verdicts | n | mean score gap | mean abs length delta |
|---|---|---|---|
| consistent across both orders | 84 | 1.786 | 361 chars |
| flipped when order swapped | 65 | **0.285** | 133 chars |

Phase 1 compared two deliberately different prompting strategies. The arena compares a
model with its own fine-tune, and they are close. A judge that stayed equally decisive
as the arms converged would not be tracking quality at all. Those flips are recorded
as ties rather than resolved, so they widen the interval instead of corrupting the
number.

### Not degeneracy

2 of 149 tuned responses are under 60 characters (1 of 149 for the baseline) and none
are empty. The model became genuinely terser, not broken. That matters for the
reading: this is a real behaviour change that the judge rewards, not a collapse that
happens to score well.

### The honest summary

DPO on 344 judge-labelled preference pairs moved a 0.5B support agent from 3.05 to
3.48 on a 9-point rubric, and roughly four fifths of that came from writing less. The
blueprint named this failure mode in advance — "reward hacking toward short-but-unhelpful
responses is a classic one" — and it is what happened, in a mild form: the responses are
shorter and slightly *more* correct, not shorter and emptier.

The cause is visible in the data rather than mysterious. The `rushed` candidate
strategy was verbose by construction and lost two thirds of the time, so brevity and
preference were correlated in the training pairs before any training ran. The rubric
named length as a non-criterion, which governs how the judge *scores* but cannot
decorrelate what the two strategies *produce*. Fixing this needs a change at the data
level — length-matched candidate strategies, or a length penalty in pair selection —
not a better rubric.

### What would make the correctness result conclusive

Roughly a third of the training pairs carry no correctness signal at all: the two
sides score identically on correctness in 31% and both score zero in 26%. A 0.5B model
answering policy questions is wrong most of the time either way, which caps how much
correctness gradient exists to extract. The realistic paths are a larger base model, a
best-of-n candidate scheme that surfaces occasional correct responses, or many more
pairs — not more epochs on this data.
