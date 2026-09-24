"""The knowledge agent always ends with visible text and a final_state."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from backend.agents import knowledge_agent


class _Client:
    def __init__(self, create):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


def _run(monkeypatch, create):
    monkeypatch.setattr(knowledge_agent, "get_llm_client", lambda agent_type: _Client(create))

    async def collect():
        return [e async for e in knowledge_agent.run_knowledge_agent("whats a soft sensor")]

    events = asyncio.run(collect())
    text = "".join(e["data"] for e in events if e["event"] == "token")
    return events, text


def _stream(*pieces, fail=False):
    async def gen():
        for p in pieces:
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=p))])
        if fail:
            raise ConnectionError("dropped")

    async def create(**_):
        return gen()

    return create


def test_all_providers_failing_is_reported_not_silent(monkeypatch):
    async def create(**_):
        raise RuntimeError("No LLM providers succeeded.")

    events, text = _run(monkeypatch, create)
    assert "couldn't reach any AI model" in text
    assert events[-1]["event"] == "final_state"


def test_empty_stream_is_reported(monkeypatch):
    events, text = _run(monkeypatch, _stream())
    assert "empty answer" in text
    assert events[-1]["event"] == "final_state"


def test_dropped_stream_keeps_partial_answer(monkeypatch):
    events, text = _run(monkeypatch, _stream("A soft sensor ", "infers", fail=True))
    assert text.startswith("A soft sensor infers")
    assert "cut off" in text
    assert events[-1]["event"] == "final_state"


def test_normal_answer_is_untouched(monkeypatch):
    _, text = _run(monkeypatch, _stream("A soft sensor ", "infers quality."))
    assert text == "A soft sensor infers quality."
