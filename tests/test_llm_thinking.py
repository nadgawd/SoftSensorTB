"""Thinking-mode request shaping for the local model and cloud fallbacks."""

from __future__ import annotations

import pytest

from backend.agents import llm_client

MESSAGES = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "", "reasoning_content": "hmm", "tool_calls": []},
    {"role": "tool", "content": "{}", "tool_call_id": "c1"},
]


@pytest.fixture
def thinking(monkeypatch):
    def configure(enabled):
        if enabled:
            monkeypatch.setenv("LOCAL_LLM_THINKING", "1")
        else:
            monkeypatch.delenv("LOCAL_LLM_THINKING", raising=False)
    return configure


def test_thinking_off_is_sent_explicitly(thinking):
    thinking(True)
    call = llm_client._local_call_kwargs({"messages": MESSAGES, "max_tokens": 4096, "temperature": 0.1}, think=False)
    assert call["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert call["max_tokens"] == 4096 and call["temperature"] == 0.1
    # Qwen3.5's template needs earlier reasoning to replay the current turn.
    assert call["messages"][1]["reasoning_content"] == "hmm"


def test_thinking_on_uses_model_card_sampling_and_room_to_think(thinking):
    thinking(True)
    original = {"messages": MESSAGES, "max_tokens": 4096, "temperature": 0.1,
                "extra_body": {"chat_template_kwargs": {"other": 1}}}
    call = llm_client._local_call_kwargs(original, think=True)
    assert call["extra_body"]["chat_template_kwargs"] == {"other": 1, "enable_thinking": True}
    assert call["extra_body"]["top_k"] == 20
    assert call["temperature"] == 0.6 and call["top_p"] == 0.95
    assert call["max_tokens"] == 16384
    assert original["max_tokens"] == 4096
    assert original["extra_body"] == {"chat_template_kwargs": {"other": 1}}


def test_model_without_thinking_gets_a_plain_request(thinking):
    thinking(False)
    call = llm_client._local_call_kwargs({"messages": MESSAGES, "max_tokens": 4096}, think=True)
    assert "extra_body" not in call
    assert call["max_tokens"] == 4096
    assert all("reasoning_content" not in m for m in call["messages"])


def test_strip_reasoning_leaves_other_fields_and_input_alone():
    call = llm_client._strip_reasoning({"messages": MESSAGES, "model": "m"})
    assert call["messages"][1] == {"role": "assistant", "content": "", "tool_calls": []}
    assert call["model"] == "m"
    assert MESSAGES[1]["reasoning_content"] == "hmm"
    untouched = {"messages": [{"role": "user", "content": "x"}]}
    assert llm_client._strip_reasoning(untouched) is untouched
