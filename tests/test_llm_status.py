"""The sidebar status chip must agree with what the fallback cascade will do."""

from __future__ import annotations

import asyncio

import pytest

from backend.agents import llm_client

CLOUD_KEYS = [cfg["api_key_env"] for cfg in llm_client.PROVIDERS.values()]


@pytest.fixture
def env(monkeypatch):
    for key in CLOUD_KEYS + [
        "LLM_MODE", "LOCAL_LLM_BASE_URL", "LOCAL_LLM_MODEL",
        "LOCAL_LLM_EXECUTION_MODEL", "LOCAL_LLM_API_KEY", "LOCAL_LLM_THINKING",
    ]:
        monkeypatch.delenv(key, raising=False)

    def configure(mode, *, local=True, cloud=(), served=None, thinking=False):
        monkeypatch.setenv("LLM_MODE", mode)
        if thinking:
            monkeypatch.setenv("LOCAL_LLM_THINKING", "1")
        else:
            monkeypatch.delenv("LOCAL_LLM_THINKING", raising=False)
        if local:
            monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://localhost:8888/v1")
            monkeypatch.setenv("LOCAL_LLM_EXECUTION_MODEL", "qwen2.5-7b")
        for provider in cloud:
            monkeypatch.setenv(llm_client.PROVIDERS[provider]["api_key_env"], "x")

        async def fake_probe(base_url, api_key, timeout):
            return served

        monkeypatch.setattr(llm_client, "_probe_local", fake_probe)
        llm_client.local_health.reset()
        return asyncio.run(llm_client.llm_status())

    return configure


def test_local_first_uses_hpc_when_model_is_served(env):
    status = env("local_first", cloud=["groq"], served=["qwen2.5-7b"])
    assert status["active"] == "local"
    assert status["local"]["serves_model"] is True


def test_local_first_falls_back_to_cloud_when_tunnel_is_down(env):
    status = env("local_first", cloud=["groq"], served=None)
    assert status["active"] == "cloud"
    assert status["local"]["online"] is False


def test_wrong_served_model_is_not_reported_as_local(env):
    status = env("local_first", cloud=["groq"], served=["some-other-model"])
    assert status["active"] == "cloud"
    assert status["local"]["online"] is True
    assert status["local"]["serves_model"] is False


def test_local_only_mode_with_server_down_has_no_llm(env):
    status = env("local", cloud=["groq"], served=None)
    assert status["active"] == "none"


def test_cloud_mode_skips_the_local_probe(env):
    status = env("cloud", cloud=["google"], served=["qwen2.5-7b"])
    assert status["active"] == "cloud"
    assert status["local"] is None
    assert status["cloud_providers"] == ["google"]


def test_thinking_is_offered_only_while_the_local_model_answers(env):
    assert env("local_first", cloud=["groq"], served=["qwen2.5-7b"], thinking=True)["thinking"] is True
    assert env("local_first", cloud=["groq"], served=None, thinking=True)["thinking"] is False
    assert env("local_first", cloud=["groq"], served=["qwen2.5-7b"])["thinking"] is False


def test_nothing_configured(env):
    status = env("local_first", local=False)
    assert status["active"] == "none"
