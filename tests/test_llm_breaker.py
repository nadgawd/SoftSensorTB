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


def _groq_then(monkeypatch, other="google"):
    for cfg in llm_client.PROVIDERS.values():
        monkeypatch.delenv(cfg["api_key_env"], raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setenv("GEMINI_API_KEY", "y")
    completions = llm_client.FallbackChatCompletions(
        [{"provider": "groq", "model": "openai/gpt-oss-20b"}, {"provider": other, "model": "gemini-flash-latest"}]
    )
    fakes = []
    for config in completions.clients:
        fake = FakeModel(config["provider"])
        fake.seen = []
        original = fake.create

        async def create(_fake=fake, _original=original, **kwargs):
            _fake.seen.append(kwargs)
            return await _original(**kwargs)

        config["client"] = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        fakes.append(fake)
    return completions, fakes


def test_a_request_over_groqs_per_request_limit_skips_it_without_a_call(monkeypatch):
    completions, (groq, google) = _groq_then(monkeypatch)
    big = [{"role": "user", "content": "x" * 30_000}]
    assert asyncio.run(completions.create(messages=big, max_tokens=4096)) == "google"
    assert groq.calls == 0 and google.seen[0]["max_tokens"] == 4096


def test_groq_gets_a_smaller_reply_budget_when_that_makes_the_request_fit(monkeypatch):
    completions, (groq, _google) = _groq_then(monkeypatch)
    small = [{"role": "user", "content": "x" * 15_000}]
    assert asyncio.run(completions.create(messages=small, max_tokens=4096)) == "groq"
    budget = groq.seen[0]["max_tokens"]
    assert 1024 <= budget < 4096
    assert llm_client.estimate_tokens({"messages": small}) + budget <= 8000


def test_the_groq_limit_can_be_lifted_for_a_paid_tier(monkeypatch):
    monkeypatch.setenv("GROQ_TPM_LIMIT", "0")
    completions, (groq, _google) = _groq_then(monkeypatch)
    assert asyncio.run(completions.create(messages=[{"role": "user", "content": "x" * 30_000}], max_tokens=4096)) == "groq"
    assert groq.seen[0]["max_tokens"] == 4096


def test_when_every_model_fails_each_reason_is_kept(monkeypatch):
    completions, (_groq, google) = _groq_then(monkeypatch)
    google.error = openai.RateLimitError(
        "quota exceeded", response=httpx.Response(429, request=REQUEST), body=None
    )
    with pytest.raises(llm_client.AllProvidersFailed) as info:
        asyncio.run(completions.create(messages=[{"role": "user", "content": "x" * 30_000}], max_tokens=4096))
    assert info.value.failures[0].startswith("groq openai/gpt-oss-20b: request of ~10,0")
    assert "8,000-token per-request limit" in info.value.failures[0]
    assert info.value.failures[1] == "google gemini-flash-latest: 429 quota exceeded"


def test_health_is_reprobed_once_stale(cascade, monkeypatch):
    async def scenario():
        await cascade.ask()
        await cascade.ask()
        calls_while_fresh = cascade.probe.calls
        monkeypatch.setattr(llm_client, "_FRESH_S", -1.0)
        await cascade.ask()
        return calls_while_fresh, cascade.probe.calls

    assert asyncio.run(scenario()) == (1, 2)
