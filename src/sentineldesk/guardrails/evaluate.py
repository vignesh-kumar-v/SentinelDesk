"""Scoring the guardrails against a labelled adversarial set.

The blueprint's rule for this phase is that an untested guardrail is worse than no
guardrail bullet at all, so this measures rather than demonstrates. Both error
directions are reported with equal weight:

* **True-positive rate** — attacks caught. A rail that misses these is decoration.
* **False-positive rate** — ordinary support tickets wrongly refused. A rail that
  blocks everything scores a perfect TPR and would take a support queue offline, so
  quoting TPR alone is how a useless rail gets shipped.

The benign cases are deliberately awkward: furious customers, cancellation threats,
requests support must decline, and innocent uses of the words "instructions" and
"system". Those are where an over-eager rail actually does damage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..config import Paths
from ..logging_utils import get_logger
from .pii import scan
from .rails import Guardrails

log = get_logger(__name__)

ADVERSARIAL = Paths.configs / "guardrails" / "adversarial.yaml"


@dataclass
class RailScore:
    name: str
    tp: int = 0
    fn: int = 0
    tn: int = 0
    fp: int = 0
    misses: list[dict] = field(default_factory=list)

    @property
    def tpr(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def fpr(self) -> float:
        return self.fp / (self.fp + self.tn) if (self.fp + self.tn) else 0.0

    def as_dict(self) -> dict:
        return {
            "rail": self.name,
            "attacks": self.tp + self.fn,
            "caught": self.tp,
            "missed": self.fn,
            "true_positive_rate": round(self.tpr, 4),
            "benign": self.tn + self.fp,
            "wrongly_blocked": self.fp,
            "false_positive_rate": round(self.fpr, 4),
            "failures": self.misses,
        }


def evaluate(guards: Guardrails, path: Path = ADVERSARIAL) -> dict:
    cases = yaml.safe_load(path.read_text(encoding="utf-8"))

    inp = RailScore("input")
    for c in cases["input_should_block"]:
        d = guards.check_input(c["text"])
        if d.allowed:
            inp.fn += 1
            inp.misses.append({"id": c["id"], "kind": "missed attack", "text": c["text"][:90]})
        else:
            inp.tp += 1
    for c in cases["input_should_allow"]:
        d = guards.check_input(c["text"])
        if d.allowed:
            inp.tn += 1
        else:
            inp.fp += 1
            inp.misses.append(
                {"id": c["id"], "kind": "blocked a real ticket", "text": c["text"][:90]}
            )

    out = RailScore("output")
    for c in cases["output_should_block"]:
        d = guards.check_output(c["text"])
        if d.allowed:
            out.fn += 1
            out.misses.append({"id": c["id"], "kind": "missed unsafe reply", "text": c["text"][:90]})
        else:
            out.tp += 1
    for c in cases["output_should_allow"]:
        d = guards.check_output(c["text"])
        if d.allowed:
            out.tn += 1
        else:
            out.fp += 1
            out.misses.append(
                {"id": c["id"], "kind": "blocked a correct reply", "text": c["text"][:90]}
            )

    pii = RailScore("pii")
    for c in cases["pii_cases"]:
        expected = set(c.get("expect") or [])
        got = set(scan(c["text"]))
        if expected:
            # Exact-set match: finding a card but missing the email is a partial
            # failure and is scored as a failure, because the un-redacted half still
            # leaves the pipeline.
            if got == expected:
                pii.tp += 1
            else:
                pii.fn += 1
                pii.misses.append({
                    "id": c["id"], "kind": "wrong PII labels",
                    "expected": sorted(expected), "got": sorted(got),
                })
        elif got:
            pii.fp += 1
            pii.misses.append({
                "id": c["id"], "kind": "redacted a non-identifier",
                "got": sorted(got), "text": c["text"][:90],
            })
        else:
            pii.tn += 1

    scores = [inp, out, pii]
    total_attacks = sum(s.tp + s.fn for s in scores)
    total_caught = sum(s.tp for s in scores)
    total_benign = sum(s.tn + s.fp for s in scores)
    total_fp = sum(s.fp for s in scores)

    return {
        "rail_model_available": guards.rails is not None,
        "cases": total_attacks + total_benign,
        "overall_true_positive_rate": round(total_caught / total_attacks, 4) if total_attacks else 0.0,
        "overall_false_positive_rate": round(total_fp / total_benign, 4) if total_benign else 0.0,
        "by_rail": [s.as_dict() for s in scores],
    }
