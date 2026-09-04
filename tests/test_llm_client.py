"""The shared chat client: JSON recovery, retry behaviour, and result plumbing."""

from __future__ import annotations

import httpx
import pytest

from sentineldesk.llm import ChatClient, LLMError, extract_json


# ------------------------------------------------------------------ json recovery
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"winner": "A"}', {"winner": "A"}),
        ('```json\n{"winner": "A"}\n```', {"winner": "A"}),
        ('```\n{"winner": "A"}\n```', {"winner": "A"}),
        ('Sure! {"winner": "A"} hope that helps', {"winner": "A"}),
        ('  \n {"winner": "A"}  ', {"winner": "A"}),
    ],
)
def test_extract_json_recovers_from_the_usual_wrappers(text, expected):
    """Models wrap JSON in prose and fences often enough that a bare loads() throws
    away perfectly good verdicts."""
    assert extract_json(text) == expected


@pytest.mark.parametrize("text", ["", "no json here", "{not: valid}", "[1, 2, 3]"])
def test_extract_json_returns_none_rather_than_guessing(text):
    assert extract_json(text) is None


def test_extract_json_prefers_the_outermost_object():
    got = extract_json('prefix {"a": {"b": 1}, "c": 2} suffix')
    assert got == {"a": {"b": 1}, "c": 2}


# ------------------------------------------------------------------ transport
def _client(handler, **kw) -> ChatClient:
    c = ChatClient("http://test/v1", "k", "m", max_retries=kw.pop("max_retries", 3), **kw)
    c._client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test/v1")
    return c


def _ok(content: str, *, finish: str = "stop", reasoning: str = "") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "m",
            "choices": [
                {"message": {"content": content, "reasoning": reasoning}, "finish_reason": finish}
            ],
            "usage": {"prompt_tokens": 7, "completion_tokens": 11},
        },
    )


def test_chat_returns_text_usage_and_finish_reason():
    res = _client(lambda r: _ok("hello", finish="length")).chat([{"role": "user", "content": "hi"}])
    assert res.text == "hello"
    assert res.prompt_tokens == 7 and res.completion_tokens == 11
    assert res.finish_reason == "length"
    assert res.latency_s >= 0


def test_reasoning_is_captured_but_kept_out_of_the_text():
    res = _client(lambda r: _ok("answer", reasoning="lots of thinking")).chat([])
    assert res.text == "answer"
    assert res.reasoning == "lots of thinking"


def test_a_server_error_is_retried_and_can_succeed():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, text="upstream is unhappy")
        return _ok("recovered")

    assert _client(handler, max_retries=4).chat([]).text == "recovered"
    assert calls["n"] == 3


def test_retries_are_bounded_and_then_it_raises():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(500, text="always broken")

    with pytest.raises(LLMError, match="failed after 3 attempts"):
        _client(handler, max_retries=3).chat([])
    assert calls["n"] == 3


def test_rate_limiting_is_retried_not_raised_immediately():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return _ok("fine") if calls["n"] > 1 else httpx.Response(429, text="slow down")

    assert _client(handler, max_retries=3).chat([]).text == "fine"


def test_chat_json_retries_the_model_on_unparseable_output():
    replies = iter(["not json at all", '{"winner": "B"}'])

    obj, res = _client(lambda r: _ok(next(replies))).chat_json([], required_keys=("winner",))
    assert obj == {"winner": "B"}
    assert res.text == '{"winner": "B"}'


def test_chat_json_rejects_json_that_is_missing_a_required_key():
    with pytest.raises(LLMError, match="no valid JSON"):
        _client(lambda r: _ok('{"reason": "no winner field"}')).chat_json(
            [], required_keys=("winner",), json_retries=2
        )


def test_chat_json_raises_the_last_reply_in_the_message():
    """The failing text has to reach the log, or a bad rubric is undebuggable."""
    with pytest.raises(LLMError, match="distinctive garbage"):
        _client(lambda r: _ok("distinctive garbage")).chat_json([], json_retries=1)


def test_map_chat_preserves_input_order():
    def handler(request):
        import json as _json

        return _ok(_json.loads(request.content)["messages"][0]["content"])

    client = _client(handler)
    batches = [[{"role": "user", "content": str(i)}] for i in range(8)]
    assert [r.text for r in client.map_chat(batches, concurrency=4)] == [str(i) for i in range(8)]
