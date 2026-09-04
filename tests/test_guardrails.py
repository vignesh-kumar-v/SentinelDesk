"""Deterministic guardrails: PII detection and response-schema validation.

The LLM rails are scored separately by `sentineldesk guardrails-eval` against the
labelled adversarial set, because they need a live model. These are the checks that
must be exactly right every time, so they are pinned here.
"""

from __future__ import annotations

import pytest

from sentineldesk.guardrails.pii import redact, scan
from sentineldesk.guardrails.schema import validate_response


# ------------------------------------------------------------------ PII
@pytest.mark.parametrize(
    ("text", "labels"),
    [
        ("reach me at jo.smith@example.co.uk", ["EMAIL"]),
        ("card 4539 1488 0343 6467 was charged", ["CARD"]),
        ("SSN 123-45-6789", ["SSN"]),
        ("call 555-123-4567", ["PHONE"]),
        ("server at 192.168.1.44", ["IPV4"]),
        ("IBAN GB29NWBK60161331926819", ["IBAN"]),
        ("token sk-live-a1b2c3d4e5f6g7h8", ["API_KEY"]),
        ("api_key_1234567890abcdef expired", ["API_KEY"]),
    ],
)
def test_scan_finds_each_identifier_type(text, labels):
    assert sorted(scan(text)) == sorted(labels)


def test_luhn_check_spares_order_references():
    """A 16-digit order id is not a card number, and redacting it destroys information
    the agent needs to do its job."""
    assert scan("order reference 8842119377445533 needs checking") == {}
    assert "CARD" in scan("card 4539 1488 0343 6467")


@pytest.mark.parametrize(
    "text",
    ["order NB-88412 shipped", "key-value pairs in the config", "sk-short",
     "ticket #48291 from last Tuesday", "we have 40 seats on the plan"],
)
def test_scan_does_not_fire_on_ordinary_support_text(text):
    """False positives here corrupt the ticket the agent has to answer."""
    assert scan(text) == {}


def test_redaction_is_typed_so_a_human_agent_still_knows_what_was_there():
    r = redact("card 4539 1488 0343 6467 and jo@example.com")
    assert "[CARD_REDACTED]" in r.text
    assert "[EMAIL_REDACTED]" in r.text
    assert "4539" not in r.text and "jo@example.com" not in r.text
    assert r.found == {"EMAIL": 1, "CARD": 1}
    assert r.redacted


def test_redaction_leaves_clean_text_untouched():
    r = redact("Refunds land in 5-7 business days.")
    assert r.text == "Refunds land in 5-7 business days."
    assert not r.redacted


def test_redaction_handles_repeated_identifiers():
    r = redact("mail a@b.com or c@d.com")
    assert r.found["EMAIL"] == 2
    assert "@" not in r.text.replace("[EMAIL_REDACTED]", "")


# ------------------------------------------------------------------ schema
def test_a_good_response_passes():
    assert validate_response(
        "Refunds land in 5-7 business days on the original payment method."
    ).ok


def test_invented_policy_ids_are_caught():
    """A citation to a policy that does not exist reads as authoritative and is
    unfalsifiable to a customer."""
    issues = validate_response("Per [BIL-09] you can refund any time you like.")
    assert issues.invented_policy_ids == ["BIL-09"]
    assert not issues.ok


def test_real_policy_ids_are_not_flagged():
    assert validate_response(
        "As set out in [BIL-01], refunds are available within 30 days of the charge."
    ).ok


def test_scaffold_leakage_is_caught():
    issues = validate_response(
        "POLICY EXCERPTS [BIL-01] Refund window. CUSTOMER TICKET Subject: refund please"
    )
    assert issues.leaks_scaffold and not issues.ok


def test_empty_and_truncated_responses_are_caught():
    assert validate_response("").empty
    assert validate_response("I will process that and then we can").truncated
    assert validate_response("Sure").too_short
