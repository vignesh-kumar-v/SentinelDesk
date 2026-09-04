"""Structured-output validation for resolution drafts.

Structure is not a judgement call, so this is pydantic and regex rather than a model.
The checks are the ones a support response can fail mechanically: empty, truncated
mid-sentence, leaking the prompt scaffolding it was given, or citing a policy id that
does not exist.

The last one is the useful one. A response citing "[BIL-09]" looks authoritative and
is unfalsifiable to a customer, and it is exactly the kind of thing a small model
produces when it pattern-matches the format of the excerpts it was shown.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from ..data.kb import load_policies

_POLICY_CITE = re.compile(r"\[([A-Z]{3}-\d{2})\]")
_SCAFFOLD = re.compile(
    r"(POLICY EXCERPTS|CUSTOMER TICKET|Write the reply to send|You are a support agent for Nimbus)",
    re.IGNORECASE,
)


class ResponseIssues(BaseModel):
    empty: bool = False
    too_short: bool = False
    truncated: bool = False
    leaks_scaffold: bool = False
    invented_policy_ids: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (
            self.empty
            or self.too_short
            or self.truncated
            or self.leaks_scaffold
            or self.invented_policy_ids
        )

    def reasons(self) -> list[str]:
        out = []
        if self.empty:
            out.append("response is empty")
        if self.too_short:
            out.append("response is too short to be an answer")
        if self.truncated:
            out.append("response ends mid-sentence")
        if self.leaks_scaffold:
            out.append("response repeats the prompt scaffolding")
        if self.invented_policy_ids:
            out.append(f"cites policy ids that do not exist: {', '.join(self.invented_policy_ids)}")
        return out


def validate_response(text: str, *, min_words: int = 8) -> ResponseIssues:
    stripped = text.strip()
    if not stripped:
        return ResponseIssues(empty=True)

    known = {p.id for p in load_policies()}
    cited = set(_POLICY_CITE.findall(stripped))

    return ResponseIssues(
        empty=False,
        too_short=len(stripped.split()) < min_words,
        # A reply that stops without terminal punctuation and without a list marker is
        # almost always a max_tokens cutoff rather than a stylistic choice.
        truncated=not stripped.endswith((".", "!", "?", ":", ")", "\"", "'"))
        and not stripped.rstrip().endswith(tuple("0123456789")),
        leaks_scaffold=bool(_SCAFFOLD.search(stripped)),
        invented_policy_ids=sorted(cited - known),
    )
