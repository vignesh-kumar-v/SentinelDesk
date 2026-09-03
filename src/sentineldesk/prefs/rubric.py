"""The judge rubric. Versioned, because every downstream number depends on it.

Design notes worth defending:

* Policy correctness dominates by construction, not by weighting alone. A response
  that contradicts the excerpts loses outright. Without that override an LLM judge
  reliably prefers the better-written wrong answer, and DPO trained on those pairs
  learns confident tone rather than accuracy.
* Length is called out explicitly as a non-criterion. LLM judges have a documented
  verbosity bias; since one of this project's two candidate strategies is verbose by
  construction, leaving that bias unaddressed would make the preference labels
  mostly a measurement of which prompt was wordier.
* The scale is coarse (0-3 / 0-2). Fine-grained scales invite the judge to
  manufacture distinctions it cannot actually make, which shows up as noise in the
  self-consistency check rather than as signal.
* The same rubric text is used for Phase 1 labelling and the Phase 5 arena. Scoring
  the benchmark with a different rubric than the one trained on would measure the
  rubric change, not the fine-tune.
"""

from __future__ import annotations

RUBRIC_VERSION = "v1"

RUBRIC = """Score each response on four dimensions.

1. POLICY CORRECTNESS (0-3) - the dimension that decides the comparison.
   3 = every factual claim matches the policy excerpts; if the customer stated a
       wrong assumption, the response corrects it plainly.
   2 = consistent with the excerpts but leaves a relevant condition or limit out.
   1 = vague enough to avoid being wrong, but does not actually answer.
   0 = contradicts an excerpt, invents policy that is not in the excerpts, agrees
       with a wrong customer assumption, or promises something the excerpts forbid.

2. COMPLETENESS (0-2)
   2 = answers everything asked, including any side question, and gives the concrete
       next step the customer should take.
   1 = answers the main question but omits the next step or a side question.
   0 = leaves the main question unanswered.

3. CONCISENESS (0-2)
   2 = leads with the answer; no filler, no restating the ticket back.
   1 = some padding or a slow opening, but readable.
   0 = buries the answer in preamble, repeats itself, or rambles.

4. TONE (0-2)
   2 = professional, addresses the customer directly, matches the urgency.
   1 = serviceable but generic or slightly off-register.
   0 = robotic, dismissive, or inappropriate.

DECIDING THE WINNER
- If exactly one response scores 0 on POLICY CORRECTNESS, the other one wins. This
  overrides every other dimension: a well-written wrong answer is worse than a
  clumsy right one, because a wrong answer is what actually costs the customer.
- Otherwise the higher total (max 9) wins.
- Call it a tie only if the totals are equal AND correctness is equal.

NOT CRITERIA - do not let these influence the verdict:
- Length. Longer is not better and shorter is not better. Judge density, not size.
- Formatting flourishes: bullet points, bold text, headers, sign-offs.
- Which response is shown first."""

JUDGE_SYSTEM = (
    "You are a strict support-quality evaluator. You compare two candidate replies to "
    "a customer ticket against internal policy excerpts, and you output only JSON. "
    "You are evaluating factual fidelity to the excerpts first and writing quality "
    "second."
)

JUDGE_TEMPLATE = """{rubric}

POLICY EXCERPTS (the ground truth; the customer has not seen these)
{policies}

CUSTOMER TICKET
Subject: {subject}
{body}

RESPONSE A
{response_a}

RESPONSE B
{response_b}

Score both, then decide. Return only this JSON and nothing else:
{{"a": {{"correctness": 0, "completeness": 0, "conciseness": 0, "tone": 0}},
  "b": {{"correctness": 0, "completeness": 0, "conciseness": 0, "tone": 0}},
  "winner": "A" | "B" | "tie",
  "reason": "one sentence, naming the deciding policy point"}}"""

DIMENSIONS = ("correctness", "completeness", "conciseness", "tone")
MAX_SCORES = {"correctness": 3, "completeness": 2, "conciseness": 2, "tone": 2}


def total_score(scores: dict[str, float]) -> float:
    return sum(float(scores.get(d, 0)) for d in DIMENSIONS)


def rubric_fingerprint() -> str:
    """Hash of the exact rubric text, recorded alongside every label.

    If the rubric is edited, labels made under the old text are no longer
    comparable with new ones, and this is what makes that detectable later.
    """
    import hashlib

    return hashlib.sha256((RUBRIC + JUDGE_SYSTEM + JUDGE_TEMPLATE).encode()).hexdigest()[:12]
