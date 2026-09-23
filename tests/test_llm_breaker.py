"""The HPC model is optional online: an outage must cost one fast failure, not every chat."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import openai
import pytest

from backend.agents import llm_client

LOCAL, CLOUD = "qwen3.5-27b", "llama-3.3-70b-versatile"
REQUEST = httpx.Request("POST", "https://hpc.example.ts.net/v1/chat/completions")


class FakeModel:
    def __init__(self, name):
        self.name = name
        self.calls = 0
        self.error = None

    async def create(self, **kwargs):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.name


@pytest.fixture
def cascade(monkeypatch):
    """A local-first cascade whose models and ``/models`` probe the test controls."""
    for cfg in llm_client.PROVIDERS.values():
        monkeypatch.delenv(cfg["api_key_env"], raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "https://hpc.example.ts.net/v1")
    monkeypatch.setenv("LOCAL_LLM_API_KEY", "k")
    monkeypatch.setenv("LOCAL_LLM_EXECUTION_MODEL", LOCAL)
    monkeypatch.delenv("LOCAL_LLM_THINKING", raising=False)

    probe = SimpleNamespace(served=[LOCAL], calls=0)

    async def fake_probe(base_url, api_key, timeout):
        probe.calls += 1
        return probe.served

    monkeypatch.setattr(llm_client, "_probe_local", fake_probe)
    llm_client.local_health.reset()

    completions = llm_client.FallbackChatCompletions(
        [{"provider": "local_vllm", "model": LOCAL}, {"provider": "groq", "model": CLOUD}]
    )
    local, cloud = FakeModel("local"), FakeModel("cloud")
    for config, fake in zip(completions.clients, (local, cloud)):
        config["client"] = SimpleNamespace(chat=SimpleNamespace(completions=fake))

    async def ask():
        return await completions.create(messages=[{"role": "user", "content": "hi"}])

    yield SimpleNamespace(ask=ask, local=local, cloud=cloud, probe=probe)
    llm_client.local_health.reset()


def test_local_client_fails_fast_and_does_not_retry(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "https://hpc.example.ts.net/v1")
    completions = llm_client.FallbackChatCompletions([{"provider": "local_vllm", "model": LOCAL}])
    client = completions.clients[0]["client"]
    assert client.max_retries == 0
    assert client.timeout.connect <= 5


def test_healthy_local_model_answers(cascade):
    assert asyncio.run(cascade.ask()) == "local"
    assert cascade.cloud.calls == 0


def test_outage_opens_the_breaker_and_recovery_closes_it(cascade):
    async def scenario():
        cascade.local.error = openai.APIConnectionError(request=REQUEST)
        first = await cascade.ask()
        assert llm_client.local_health.down

        # While open, chats go straight to the cloud without touching the tunnel.
        second = await cascade.ask()
        assert cascade.local.calls == 1

        cascade.local.error = None
        await llm_client.local_health.probe()
        third = await cascade.ask()
        return first, second, third

    assert asyncio.run(scenario()) == ("cloud", "cloud", "local")


def test_unreachable_tunnel_is_skipped_without_a_chat_call(cascade):
    cascade.probe.served = None
    assert asyncio.run(cascade.ask()) == "cloud"
    assert cascade.local.calls == 0
    assert llm_client.local_health.down


def test_wrong_api_key_counts_as_an_outage(cascade):
    cascade.local.error = openai.AuthenticationError(
        "bad key", response=httpx.Response(401, request=REQUEST), body=None
    )
    assert asyncio.run(cascade.ask()) == "cloud"
    assert llm_client.local_health.down


def test_a_bad_request_does_not_open_the_breaker(cascade):
    cascade.local.error = openai.BadRequestError(
        "context too long", response=httpx.Response(400, request=REQUEST), body=None
    )
    assert asyncio.run(cascade.ask()) == "cloud"
    assert not llm_client.local_health.down


def test_health_is_reprobed_once_stale(cascade, monkeypatch):
    async def scenario():
        await cascade.ask()
        await cascade.ask()
        calls_while_fresh = cascade.probe.calls
        monkeypatch.setattr(llm_client, "_FRESH_S", -1.0)
        await cascade.ask()
        return calls_while_fresh, cascade.probe.calls

    assert asyncio.run(scenario()) == (1, 2)
