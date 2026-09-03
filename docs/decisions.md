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
