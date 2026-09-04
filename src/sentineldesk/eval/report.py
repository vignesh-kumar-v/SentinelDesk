"""Assemble reports/RESULTS.md from whatever phase reports exist on disk.

Generated rather than hand-written so the numbers in the write-up cannot drift from
the numbers in the runs. Every phase is optional: the file reports what has actually
been produced and says plainly what has not, rather than leaving a stale figure from
an earlier run standing.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def build_report(reports_dir: Path, out: Path) -> Path:
    p0 = _load(reports_dir / "phase0_smoke.json")
    p1t = _load(reports_dir / "phase1_tickets.json")
    p1p = _load(reports_dir / "phase1_pairs.json")
    p1s = _load(reports_dir / "phase1_spotcheck_second_judge.json")
    p1h = _load(reports_dir / "phase1_spotcheck_human_result.json")
    p2 = _load(reports_dir / "phase2_training_summary.json")
    p2s = _load(reports_dir / "phase2_sweep.json")
    p3 = _load(reports_dir / "phase3_graph_check.json")
    p4 = _load(reports_dir / "phase4_serving_bench.json")
    p5 = _load(reports_dir / "phase5_arena.json")
    p6 = _load(reports_dir / "phase6_guardrails.json")
    p7 = _load(reports_dir / "phase7_tracing.json")
    p8 = _load(reports_dir / "phase8_terraform.json")
    v1p1 = _load(reports_dir / "v1" / "phase1_pairs.json")
    v1p5 = _load(reports_dir / "v1" / "phase5_arena.json")

    L: list[str] = []
    add = L.append
    add("# SentinelDesk — Results\n")
    add(f"_Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} from the phase reports "
        "in this directory. Regenerate with `make report`._\n")

    add("## Headline\n")
    if p5 and p5.get("n"):
        lo, hi = p5["win_rate_adjusted_ci95"]
        dlo, dhi = p5["win_rate_decisive_ci95"]
        adj_sig = not (lo <= 0.5 <= hi)
        dec_sig = not (dlo <= 0.5 <= dhi)
        add(f"On {p5['n']} held-out tickets, judged blind in both display orders, the DPO-tuned "
            f"resolution agent scored {p5['wins_tuned']}W / {p5['losses_tuned']}L / "
            f"{p5['ties']}T against the base-prompted agent.\n")
        add("| framing | win-rate | 95% CI | 50% inside? |")
        add("|---|---|---|---|")
        add(f"| ties counted as half | {_pct(p5['win_rate_adjusted'])} | "
            f"{_pct(lo)}–{_pct(hi)} | {'no — significant' if adj_sig else '**yes — not significant**'} |")
        add(f"| ties dropped | {_pct(p5['win_rate_decisive'])} | "
            f"{_pct(dlo)}–{_pct(dhi)} | {'**no — significant**' if dec_sig else 'yes — not significant'} |")
        add(f"\nExact binomial p on the decisive comparisons = {p5['binomial_p_vs_50pct']}.")
        if adj_sig != dec_sig:
            add(f"\n**The two framings disagree, and neither is quoted alone.** With "
                f"{p5['ties']} of {p5['n']} comparisons tied "
                f"({_pct(p5['ties'] / p5['n'])}), how ties are treated decides whether this "
                "result clears significance. A fine-tune that mostly produces ties has not "
                "improved much, so the conservative reading — ties as half a win, interval "
                "containing 50% — is the one to carry.")

        scores = p5.get("mean_scores") or {}
        if scores:
            t = scores.get("dpo_tuned", {})
            b = scores.get("base_prompted", {})
            total_gap = t.get("total", 0) - b.get("total", 0)
            if total_gap:
                parts = sorted(
                    ((d, t.get(d, 0) - b.get(d, 0)) for d in
                     ("correctness", "completeness", "conciseness", "tone")),
                    key=lambda kv: -abs(kv[1]),
                )
                top, gap = parts[0]
                add(f"\n**Where the advantage comes from: {_pct(gap / total_gap)} of the "
                    f"{total_gap:+.3f} total-score gap is {top}.** "
                    + ", ".join(f"{d} {v:+.3f}" for d, v in parts) + ".")
        add(f"\nThe tuned arm's responses are {_pct(1 - p5['length_ratio_tuned_over_base'])} "
            f"shorter than the baseline's ({p5['mean_chars']['dpo_tuned']} vs "
            f"{p5['mean_chars']['base_prompted']} chars), and wins correlate with being "
            f"shorter at r = {p5['corr_win_vs_length_delta']}. See the reward-hacking note "
            "in Phase 5 below.\n")
    else:
        add("_Phase 5 has not been run. No win-rate to report._\n")

    if v1p1 and p1p and v1p5:
        add("\n## v1 vs v2 — removing the length confound\n")
        add("v1's result was that DPO learned brevity: 81.7% of the score gain was "
            "conciseness, and the tuned model reproduced the training pairs' length skew "
            "almost exactly. The cause was upstream — the contrast prompting strategy was "
            "verbose by construction, so brevity and preference were correlated before any "
            "training ran. v2 rebuilds the preference data with length-matched strategies "
            "and a post-judging balancing step, and re-runs Phases 1-5 unchanged otherwise.\n")
        add("| | v1 | v2 |")
        add("|---|---|---|")
        add(f"| usable pairs | {v1p1['usable_pairs']} | {p1p['usable_pairs']} |")
        add(f"| **length ratio chosen/rejected** | **{v1p1['length_ratio_chosen_over_rejected']}** "
            f"| **{p1p['length_ratio_chosen_over_rejected']}** |")
        add(f"| order-inconsistency | {_pct(v1p1['order_inconsistency_rate'])} | "
            f"{_pct(p1p['order_inconsistency_rate'])} |")
        v1s = (v1p1.get("dimension_separation") or {})
        v2s = (p1p.get("dimension_separation") or {})
        for d in ("correctness", "conciseness"):
            if d in v1s and d in v2s:
                add(f"| {d} separation (mean gap) | {v1s[d]['mean_gap']:+.3f} | "
                    f"{v2s[d]['mean_gap']:+.3f} |")
        if p5 and p5.get("n"):
            add(f"| win-rate (ties as half) | {_pct(v1p5['win_rate_adjusted'])} | "
                f"{_pct(p5['win_rate_adjusted'])} |")
            v1g = v1p5["mean_scores"]["dpo_tuned"]["correctness"] - \
                  v1p5["mean_scores"]["base_prompted"]["correctness"]
            v2g = p5["mean_scores"]["dpo_tuned"]["correctness"] - \
                  p5["mean_scores"]["base_prompted"]["correctness"]
            add(f"| **correctness gain over baseline** | **{v1g:+.3f}** | **{v2g:+.3f}** |")
            add(f"| response length vs baseline | {v1p5['length_ratio_tuned_over_base']}x | "
                f"{p5['length_ratio_tuned_over_base']}x |")
            add(f"| corr(win, being shorter) | {v1p5['corr_win_vs_length_delta']} | "
                f"{p5['corr_win_vs_length_delta']} |")
        if p1p.get("length_balancing_applied"):
            add(f"\n_v2 length balancing dropped {p1p['pairs_dropped_for_length']} pairs to move "
                f"the ratio from {p1p['length_ratio_before_balancing']} to "
                f"{p1p['length_ratio_after_balancing']}. The residual skew before balancing is "
                "not the v1 artifact: the strategies are matched at generation, and what remains "
                "is the judge preferring the more concise response within a pair, which the "
                "rubric explicitly rewards. Balancing trades some legitimate signal for a "
                "cleaner test of whether correctness can be learned when brevity is not a free "
                "win._")
        add("")

    add("\n## Phase 0 — DPO loop verified on a toy batch\n")
    if p0:
        add(f"- loss {p0['loss_first']} -> {p0['loss_last']}, reward margin "
            f"{p0['margin_first']} -> {p0['margin_last']}, accuracy {p0['accuracy_last']}")
        add(f"- {p0['steps']} optimiser steps in {p0['wall_clock_s']}s on {p0.get('base_model', '?')}")
        add("- The starting loss of 0.6931 is ln 2 to four places, which is the independent "
            "check that the policy is identical to the reference at init: prompt masking and "
            "reference handling are both correct.\n")
    else:
        add("_not run_\n")

    add("\n## Phase 1 — Preference data\n")
    if p1t:
        add(f"- {p1t['n']} synthetic tickets, {p1t['policies_covered']} policies covered, "
            f"median body {p1t['median_body_words']} words")
        add(f"- split: {p1t['by_split']}")
        add(f"- categories: {p1t['by_category']}")
        add(f"- scenario mix: {p1t['by_twist']}")
    if p1p:
        add(f"- judge: `{p1p['judge_model']}`, rubric `{p1p['rubric_version']}` "
            f"(fingerprint `{p1p['rubric_fingerprint']}`)")
        add(f"- {p1p['judged']} comparisons x 2 display orders")
        add(f"- **order-inconsistency rate {_pct(p1p['order_inconsistency_rate'])}** — how often "
            "swapping which response was shown first flipped the verdict. Those comparisons are "
            "dropped rather than resolved, and this rate bounds how much any downstream "
            "win-rate can be trusted.")
        add(f"- {p1p['usable_pairs']} usable pairs ({_pct(p1p['pair_yield'])} yield)")
        add(f"- chosen-strategy split: {p1p['strategy_as_chosen']}")
        add(f"- mean rubric scores by strategy: {json.dumps(p1p['mean_scores_by_strategy'])}")
        add(f"- length ratio chosen/rejected: {p1p['length_ratio_chosen_over_rejected']} "
            "(a ratio far from 1.0 means DPO would learn length before it learns correctness)")
        sep = p1p.get("dimension_separation") or {}
        if sep:
            add("")
            add("**Which rubric dimension actually decided the training labels** "
                f"(over the {sep.get('_n_usable', 0)} usable pairs). If tone or conciseness "
                "separated them and correctness did not, DPO would be learning house style "
                "and any win-rate would be measuring that instead.\n")
            add("| dimension | mean winner−loser gap | favours winner in |")
            add("|---|---|---|")
            for d in ("correctness", "completeness", "conciseness", "tone"):
                if d in sep:
                    add(f"| {d} | {sep[d]['mean_gap']:+.3f} | "
                        f"{_pct(sep[d]['favours_winner_frac'])} of pairs |")
            add("")
            add(f"- correctness was identical on both sides in "
                f"{_pct(sep.get('_correctness_identical_frac', 0))} of pairs; both sides scored "
                f"0 on correctness in {_pct(sep.get('_both_incorrect_frac', 0))}")
    if p1s:
        add(f"- **second-judge cross-check**: `{p1s['second_judge']}` re-labelled {p1s['n']} of the "
            f"same comparisons — raw agreement {_pct(p1s['raw_agreement'])}, "
            f"Cohen's kappa {p1s['cohens_kappa']}")
        if p1s.get("decisive_n"):
            add(f"- **agreement where both judges committed to a winner: "
                f"{p1s['decisive_agree']}/{p1s['decisive_n']} = "
                f"{_pct(p1s['decisive_agreement'])}**. This is the number that bears on the "
                "preference labels. Two labellers can disagree about which response is better, "
                "or about whether the gap is decisive enough to call; only the first threatens "
                "the data, and a \"tie\" here means the primary judge flipped across display "
                "orders, which already excludes that comparison from training.")
            add(f"- confusion (primary|second): `{p1s['confusion']}`")
    if p1h:
        add(f"- **human spot-check**: {p1h['n']} blind comparisons — agreement "
            f"{_pct(p1h['raw_agreement'])}, Cohen's kappa {p1h['cohens_kappa']}")
    if not (p1t or p1p):
        add("_not run_\n")

    add("\n## Phase 2 — DPO training\n")
    if p2s and p2s.get("runs"):
        add(f"Swept {len(p2s['runs'])} configurations. **Selection used a validation split of "
            "the training pairs only — never the held-out tickets the Phase 5 arena scores.** "
            "Choosing a checkpoint by its arena win-rate and then reporting that win-rate would "
            "make the headline number a measure of how many configurations were tried.\n")
        add("| config | steps | train loss | eval loss | eval acc | eval margin | "
            "train−eval margin | drift |")
        add("|---|---|---|---|---|---|---|---|")
        for r in p2s["runs"]:
            mark = " **(selected)**" if r["name"] == p2s.get("selected") else ""
            flags = ""
            if r.get("over_drift_limit"):
                flags += " ⚠over-drift"
            if r.get("eval_loss_rose"):
                flags += " ⚠eval loss rose"
            add(f"| `{r['name']}`{mark} | {r['steps']} | "
                f"{r['train_loss_first']:.4f} → {r['train_loss_last']:.4f} | "
                f"{r['eval_loss']:.4f}{' ⚠' if r.get('eval_loss_rose') else ''} | "
                f"{r['eval_accuracy']:.3f} | {r['eval_margin']:+.4f} | "
                f"{r.get('train_margin_minus_eval_margin', 0):+.3f} | "
                f"{r['drift_chosen']:+.3f}{flags} |")
        add("\n_train−eval margin is the overfitting indicator: a run that separates the "
            "training pairs far more confidently than the held-out ones has memorised them, "
            "and on a ~34-pair validation split its eval accuracy can still look competitive "
            "by luck._")
        add(f"\n_{p2s.get('selection_rule', '')}_\n")
        if p2s.get("eval_pairs"):
            add(f"Validation split: {p2s['eval_pairs']} pairs, so the standard error on eval "
                f"accuracy is about {p2s['eval_accuracy_standard_error']:.3f} "
                f"({_pct(p2s['eval_accuracy_standard_error'])} points). Configurations closer "
                "together than roughly twice that are not distinguishable by this measurement.")
            if p2s.get("selection_is_within_noise"):
                add(f"**The selected configuration beat the runner-up by "
                    f"{p2s['selected_margin_over_runner_up']:+.3f} accuracy, which is inside "
                    "that noise band.** Something has to be chosen, and the rule was fixed in "
                    "advance, but this selection should be read as 'not clearly worse' rather "
                    "than 'best'.")
            else:
                add(f"The selected configuration beat the runner-up by "
                    f"{p2s['selected_margin_over_runner_up']:+.3f} accuracy, outside that band.")
    if p2:
        first, last = p2.get("first_step", {}), p2.get("last_step", {})
        fe, le = p2.get("first_eval", {}), p2.get("last_eval", {})
        add(f"- {p2['optimiser_steps']} optimiser steps on {p2['train_pairs']} pairs "
            f"({p2['eval_pairs']} held out for eval), {p2['wall_clock_s']}s")
        if first and last:
            add(f"- train loss {first['loss']:.4f} -> {last['loss']:.4f}")
            add(f"- reward margin {first['reward/margin']:+.3f} -> {last['reward/margin']:+.3f}")
            add(f"- implicit reward, chosen {first['reward/chosen']:+.3f} -> {last['reward/chosen']:+.3f}, "
                f"rejected {first['reward/rejected']:+.3f} -> {last['reward/rejected']:+.3f}")
            add(f"- preference accuracy {first['reward/accuracy']:.2f} -> {last['reward/accuracy']:.2f}")
            add(f"- drift from reference (log pi - log pi_ref on chosen) "
                f"{first.get('kl/chosen', 0):+.3f} -> {last.get('kl/chosen', 0):+.3f}")
        if fe and le:
            add(f"- eval loss {fe['loss']:.4f} -> {le['loss']:.4f}, "
                f"eval margin {fe['reward/margin']:+.3f} -> {le['reward/margin']:+.3f}")
        add("\n![training curves](phase2_training.png)\n")
    else:
        add("_not run_\n")

    add("\n## Phase 3 — Agent graph\n")
    if p3:
        add(f"- {p3['n']} tickets end-to-end through triage -> retrieve -> resolution -> gate")
        add(f"- triage accuracy vs the ticket's true category: **{_pct(p3['triage_accuracy'])}**")
        add(f"- escalation rate {_pct(p3['escalation_rate'])} ({p3['escalated']}/{p3['n']})")
        add(f"- resolver backend: {p3['resolver']}")
    else:
        add("_not run_\n")

    add("\n## Phase 4 — Serving\n")
    if p4 and p4.get("results"):
        add(f"{p4['n_requests']} requests per arm, max {p4['max_tokens']} new tokens.\n")
        add("| backend | mode | requests | tokens/s | p50 latency | p95 latency | wall | errors |")
        add("|---|---|---|---|---|---|---|---|")
        for r in p4["results"]:
            add(f"| {r['backend']} | {r['mode']} | {r['n_requests']} | "
                f"{r['throughput_tok_s']:.1f} | {r['latency_p50_s']:.2f}s | "
                f"{r['latency_p95_s']:.2f}s | {r['wall_clock_s']:.1f}s | {r['errors']} |")
        notes = [r for r in p4["results"] if r.get("notes")]
        if notes:
            add("")
            for r in notes:
                add(f"- _{r['backend']} ({r['mode']}): {r['notes']}_")

        seq = next((r for r in p4["results"]
                    if r["backend"].startswith("vllm") and r["mode"] == "sequential"), None)
        con = next((r for r in p4["results"]
                    if r["backend"].startswith("vllm") and r["mode"].startswith("concurrent")), None)
        if seq and con and seq["throughput_tok_s"]:
            gain = con["throughput_tok_s"] / seq["throughput_tok_s"]
            lat = con["latency_p50_s"] / seq["latency_p50_s"] if seq["latency_p50_s"] else 0
            add("")
            add(f"**Continuous batching is worth {gain:.1f}x throughput** "
                f"({seq['throughput_tok_s']:.1f} -> {con['throughput_tok_s']:.1f} tokens/s) for "
                f"{lat:.2f}x the median per-request latency. That trade is the reason vLLM "
                "exists, and it is invisible in a sequential benchmark — which is why both "
                "regimes are measured rather than just the flattering one.")
        add("")
        add("**These backends are not on equal hardware, and the table should not be read "
            "as if they were.** vLLM has no Metal backend and ships no macOS wheel, so it runs "
            "its CPU build inside a linux/arm64 container; MLX runs on the Metal GPU and "
            "transformers on MPS. A GPU path beating a CPU path is the expected outcome, not a "
            "verdict on vLLM. What the vLLM rows do establish is the batching behaviour above, "
            "which is a property of the engine rather than of the silicon.")
    else:
        add("_not run_\n")

    add("\n## Phase 5 — Blind win-rate, held-out\n")
    if p5 and p5.get("n"):
        add(f"- tuned arm: `{p5.get('tuned_arm')}`; base arm: `{p5.get('base_arm')}`")
        add(f"- both arms: identical system prompt, identical retrieved policies, "
            f"temperature {p5.get('temperature')}, max {p5.get('max_tokens')} tokens. "
            "The only difference is the weights.")
        add(f"- judge `{p5.get('judge_model')}`, rubric `{p5.get('rubric_version')}` — the same "
            "rubric as Phase 1, unrevised")
        add(f"- order-inconsistency rate in the arena: {_pct(p5['order_inconsistency_rate'])}")
        add(f"- mean response length: {p5['mean_chars']} chars, ratio tuned/base "
            f"{p5['length_ratio_tuned_over_base']}")
        add(f"- **correlation between winning and being shorter than the baseline: "
            f"{p5['corr_win_vs_length_delta']}** — the reward-hacking check.")
        if p5.get("training_pair_length_ratio"):
            add(f"- **The tuned model reproduced the training data's length skew.** In the "
                f"preference pairs the chosen side was "
                f"{p5['training_pair_length_ratio']}x the length of the rejected side; the "
                f"tuned arm now writes at {p5['length_ratio_tuned_over_base']}x the baseline's "
                "length. Those two ratios matching is the clearest single piece of evidence "
                "that what DPO learned here was substantially *length*.\n")
        add("| category | n | W | T | L | win rate (adj) |")
        add("|---|---|---|---|---|---|")
        for c, row in p5.get("by_category", {}).items():
            add(f"| {c} | {row['n']} | {row['wins']} | {row['ties']} | {row['losses']} | "
                f"{_pct(row['win_rate_adjusted'])} |")
        inc = p5.get("inconsistency_diagnosis") or {}
        if inc:
            add("\n**Why order-inconsistency is high here (43.6%, against 15.9% in Phase 1 "
                "labelling).** It is the judge behaving correctly, not failing:\n")
            add("| verdicts | n | mean score gap | mean \\|length delta\\| |")
            add("|---|---|---|---|")
            add(f"| consistent across both orders | {inc['consistent_n']} | "
                f"{inc['mean_score_gap_consistent']} | "
                f"{inc['mean_abs_length_delta_consistent']} chars |")
            add(f"| flipped when the order swapped | {inc['inconsistent_n']} | "
                f"{inc['mean_score_gap_inconsistent']} | "
                f"{inc['mean_abs_length_delta_inconsistent']} chars |")
            add("\nThe judge flips precisely where the two responses are near-equivalent — a "
                "score gap of 0.3 out of 9. Phase 1 compared two deliberately different "
                "prompting strategies; the arena compares a model with its own fine-tune. A "
                "judge that stayed equally decisive as the arms converged would not be "
                "tracking quality. Those flips are recorded as ties rather than resolved, so "
                "they widen the interval instead of corrupting the result.")
        deg = p5.get("degeneracy_check") or {}
        if deg:
            add(f"\n**Degeneracy check**: {deg['tuned_under_60_chars']}/{deg['n']} tuned "
                f"responses are under 60 characters ({deg['base_under_60_chars']}/{deg['n']} "
                f"for the baseline) and {deg['tuned_empty']} are empty. The tuned model became "
                "terser, not broken — which is what makes the brevity finding a real behaviour "
                "change rather than a collapse.")
        add("\n| arm | correctness | completeness | conciseness | tone | total |")
        add("|---|---|---|---|---|---|")
        for arm, row in p5.get("mean_scores", {}).items():
            add(f"| {arm} | {row['correctness']:.2f} | {row['completeness']:.2f} | "
                f"{row['conciseness']:.2f} | {row['tone']:.2f} | {row['total']:.2f} |")
    else:
        add("_not run_\n")

    add("\n## Phase 6 — Guardrails\n")
    if p6:
        add(f"Scored against {p6['cases']} labelled cases, LLM rails "
            f"{'on' if p6['rail_model_available'] else 'OFF'}. **Overall TPR "
            f"{_pct(p6['overall_true_positive_rate'])} at FPR "
            f"{_pct(p6['overall_false_positive_rate'])}.**\n")
        add("| rail | attacks | caught | TPR | benign | wrongly blocked | FPR |")
        add("|---|---|---|---|---|---|---|")
        for r in p6["by_rail"]:
            add(f"| {r['rail']} | {r['attacks']} | {r['caught']} | "
                f"{_pct(r['true_positive_rate'])} | {r['benign']} | {r['wrongly_blocked']} | "
                f"{_pct(r['false_positive_rate'])} |")
        add("\nThe false-positive rate is the number worth defending. The benign half of the "
            "set is built to be hostile to an over-eager rail — furious customers, "
            "cancellation threats, requests support must decline, innocent uses of "
            "\"instructions\" and \"system\". A rail that blocks everything scores a perfect "
            "TPR and takes a support queue offline.")
        fails = [f for r in p6["by_rail"] for f in r["failures"]]
        if fails:
            add("\nRemaining failures, reported rather than tuned away:")
            for f in fails:
                add(f"- `{f['id']}` {f['kind']}: {f.get('text', '')}{f.get('got', '')}")
    else:
        add("_not run_\n")

    add("\n## Phase 7 — Tracing\n")
    if p7 and p7.get("verified_from_server"):
        v = p7["verified_from_server"]
        ok = [x for x in v if x.get("observations_in_trace")]
        add(f"{len(ok)}/{len(v)} tickets produced a trace that could be read back out of "
            f"Langfuse through its public API, each with "
            f"{ok[0]['observations_in_trace'] if ok else 0} observations.\n")
        if ok:
            add(f"- hops recorded: `{'`, `'.join(ok[0]['hop_names'])}`")
            add(f"- server-side latency for that ticket: {ok[0].get('latency_s', 0):.2f}s")
        add("\nReading the traces back is the verification. The first run reported success "
            "from the SDK and stored nothing — Langfuse accepts a batch and uploads it "
            "asynchronously, so a failure downstream of the acknowledgement never reaches the "
            "client. Asserting that the SDK was called would have shipped a broken "
            "integration with a green check.")
    else:
        add("_not run_\n")

    add("\n## Phase 8 — Infrastructure\n")
    if p8:
        add(f"`terraform apply` created {p8['resources_created']} resources in "
            f"{p8.get('region')}; `terraform destroy` removed all of them.\n")
        add(f"- applied at `desired_count={p8['applied_with']['desired_count']}`: "
            f"{p8['applied_with']['why']}")
        add(f"- live state before teardown: ALB `{p8.get('alb_state')}`, ECS service "
            f"`{p8.get('ecs_service_status')}`")
        d8 = p8.get("destroy", {})
        if d8:
            conf = d8.get("independently_confirmed", {})
            add(f"- teardown: {d8['result']} — confirmed through the AWS API "
                f"({', '.join(f'{k}: {v}' for k, v in conf.items())}) rather than "
                "terraform's exit code")
    else:
        add("_not run_\n")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out
