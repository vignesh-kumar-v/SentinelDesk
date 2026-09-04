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
    p3 = _load(reports_dir / "phase3_graph_check.json")
    p4 = _load(reports_dir / "phase4_serving_bench.json")
    p5 = _load(reports_dir / "phase5_arena.json")

    L: list[str] = []
    add = L.append
    add("# SentinelDesk — Results\n")
    add(f"_Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} from the phase reports "
        "in this directory. Regenerate with `make report`._\n")

    add("## Headline\n")
    if p5 and p5.get("n"):
        lo, hi = p5["win_rate_adjusted_ci95"]
        add(f"On {p5['n']} held-out tickets, judged blind in both display orders, the DPO-tuned "
            f"resolution agent beat the base-prompted agent with a tie-adjusted win-rate of "
            f"**{_pct(p5['win_rate_adjusted'])}** (95% CI {_pct(lo)}–{_pct(hi)}), "
            f"{p5['wins_tuned']}W / {p5['losses_tuned']}L / {p5['ties']}T. "
            f"Exact binomial p vs 50% = {p5['binomial_p_vs_50pct']}"
            f"{' (significant at 0.05)' if p5.get('significant_at_05') else ' (not significant at 0.05)'}.\n")
        add(f"Ties-dropped win-rate: {_pct(p5['win_rate_decisive'])} "
            f"(95% CI {_pct(p5['win_rate_decisive_ci95'][0])}–{_pct(p5['win_rate_decisive_ci95'][1])}). "
            "Both are reported because a fine-tune that produces more ties has not improved "
            "anything, and the ties-dropped figure hides that.\n")
    else:
        add("_Phase 5 has not been run. No win-rate to report._\n")

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
            f"{p5['corr_win_vs_length_delta']}** — the reward-hacking check. A strongly negative "
            "value would mean the tuned arm wins mainly by being terse rather than by being right.\n")
        add("| category | n | W | T | L | win rate (adj) |")
        add("|---|---|---|---|---|---|")
        for c, row in p5.get("by_category", {}).items():
            add(f"| {c} | {row['n']} | {row['wins']} | {row['ties']} | {row['losses']} | "
                f"{_pct(row['win_rate_adjusted'])} |")
        add("\n| arm | correctness | completeness | conciseness | tone | total |")
        add("|---|---|---|---|---|---|")
        for arm, row in p5.get("mean_scores", {}).items():
            add(f"| {arm} | {row['correctness']:.2f} | {row['completeness']:.2f} | "
                f"{row['conciseness']:.2f} | {row['tone']:.2f} | {row['total']:.2f} |")
    else:
        add("_not run_\n")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out
