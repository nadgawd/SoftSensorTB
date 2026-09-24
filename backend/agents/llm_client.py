"""Multi-provider Fallback LLM Client."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import httpx
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    AsyncOpenAI,
    AuthenticationError,
    InternalServerError,
    RateLimitError,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv()

logger = logging.getLogger(__name__)

PROVIDERS = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "google": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
    },
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "api_key_env": "CEREBRAS_API_KEY",
    },
}

# Agent-specific model env vars when using local vLLM
_LOCAL_MODEL_ENV = {
    "execution": "LOCAL_LLM_EXECUTION_MODEL",
    "knowledge": "LOCAL_LLM_KNOWLEDGE_MODEL",
    "router": "LOCAL_LLM_ROUTER_MODEL",
}


def _llm_mode() -> str:
    return os.getenv("LLM_MODE", "cloud").strip().lower()


def local_supports_thinking() -> bool:
    """Whether the local model has a thinking mode toggled via ``enable_thinking``."""
    return os.getenv("LOCAL_LLM_THINKING", "").strip().lower() in ("1", "true", "yes")


# Qwen3.5 model-card settings for thinking mode on precise tasks.
_THINKING_SAMPLING = {"temperature": 0.6, "top_p": 0.95}
_THINKING_MIN_MAX_TOKENS = 16384


def _local_call_kwargs(kwargs: dict[str, Any], think: bool) -> dict[str, Any]:
    if not local_supports_thinking():
        return _strip_reasoning(kwargs)
    call = dict(kwargs)
    extra = dict(call.get("extra_body") or {})
    template_kwargs = dict(extra.get("chat_template_kwargs") or {})
    template_kwargs["enable_thinking"] = think
    extra["chat_template_kwargs"] = template_kwargs
    if think:
        call.update(_THINKING_SAMPLING)
        extra.setdefault("top_k", 20)
        call["max_tokens"] = max(call.get("max_tokens") or 0, _THINKING_MIN_MAX_TOKENS)
    call["extra_body"] = extra
    return call


def _strip_reasoning(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop ``reasoning_content`` from messages; strict providers reject unknown fields."""
    messages = kwargs.get("messages")
    if not messages or not any(isinstance(m, dict) and "reasoning_content" in m for m in messages):
        return kwargs
    call = dict(kwargs)
    call["messages"] = [
        {k: v for k, v in m.items() if k != "reasoning_content"} if isinstance(m, dict) else m
        for m in messages
    ]
    return call


def _local_vllm_config(agent_type: str) -> dict[str, str] | None:
    base_url = os.getenv("LOCAL_LLM_BASE_URL", "").strip()
    if not base_url:
        return None
    model_env = _LOCAL_MODEL_ENV.get(agent_type, "LOCAL_LLM_MODEL")
    model = os.getenv(model_env) or os.getenv("LOCAL_LLM_MODEL")
    if not model:
        logger.warning(
            "LOCAL_LLM_BASE_URL is set but no model for %s (%s or LOCAL_LLM_MODEL)",
            agent_type,
            model_env,
        )
        return None
    api_key = os.getenv("LOCAL_LLM_API_KEY", "EMPTY")
    return {"base_url": base_url, "api_key": api_key, "model": model}


def _openrouter_model(env: str, default: str) -> str:
    # OpenRouter's ":free" line-up changes; override without a code change.
    return os.getenv(env, "").strip() or default


# Gemini uses the "-latest" aliases: pinned versions get retired for new keys
# (gemini-2.5-* returned 404 in Sep 2026). Cerebras' free tier now answers 402.
def _cloud_cascades(agent_type: str) -> list[dict[str, str]]:
    if agent_type == "execution":
        return [
            {"provider": "groq", "model": "qwen/qwen3.8-27b"},
            {"provider": "openrouter", "model": _openrouter_model("OPENROUTER_EXECUTION_MODEL", "qwen/qwen3.8-27b:free")},
            {"provider": "openrouter", "model": "nvidia/nemotron-3-super-120b-a12b:free"},
            {"provider": "google", "model": "gemini-flash-latest"},
            {"provider": "google", "model": "gemini-flash-lite-latest"},
            {"provider": "groq", "model": "openai/gpt-oss-20b"},
        ]
    if agent_type == "knowledge":
        return [
            {"provider": "groq", "model": "qwen/qwen3.8-27b"},
            {"provider": "google", "model": "gemini-flash-latest"},
            {"provider": "google", "model": "gemini-flash-lite-latest"},
            {"provider": "openrouter", "model": _openrouter_model("OPENROUTER_KNOWLEDGE_MODEL", "google/gemma-4-31b-it:free")},
            {"provider": "groq", "model": "openai/gpt-oss-120b"},
        ]
    if agent_type == "router":
        return [
            {"provider": "groq", "model": "qwen/qwen3.8-27b"},
            {"provider": "google", "model": "gemini-flash-lite-latest"},
        ]
    return [{"provider": "groq", "model": "qwen/qwen3.8-27b"}]


def _build_cascades(agent_type: str) -> list[dict[str, str]]:
    mode = _llm_mode()
    local = _local_vllm_config(agent_type)
    cloud = _cloud_cascades(agent_type)

    if mode == "local":
        if local:
            return [{"provider": "local_vllm", "model": local["model"]}]
        logger.warning("LLM_MODE=local but local vLLM is not configured; using cloud.")
        return cloud

    if mode == "local_first" and local:
        return [{"provider": "local_vllm", "model": local["model"]}] + cloud

    return cloud


# The local server is reached through a laptop tunnel that may be off. Connect
# fails fast; reads stay long enough for a thinking-mode first token.
LOCAL_TIMEOUT = httpx.Timeout(connect=3.0, read=180.0, write=30.0, pool=10.0)
CLOUD_TIMEOUT = httpx.Timeout(120.0, connect=10.0)
_BREAKER_S = 30.0
_FRESH_S = 30.0
_PROBE_TIMEOUT_S = 3.0


def _is_local_outage(exc: Exception) -> bool:
    """Errors meaning the server is unreachable or misconfigured, not a bad request."""
    if isinstance(exc, (APIConnectionError, AuthenticationError, InternalServerError)):
        return True
    return isinstance(exc, APIStatusError) and exc.status_code >= 500


class LocalHealth:
    """Circuit breaker for the local vLLM server.

    After an outage, chat skips local and goes straight to the cloud cascade
    while a background ``/models`` probe runs every 30 s; the first successful
    probe closes the breaker. A result older than 30 s is re-probed before use,
    so a tunnel that went away is noticed in seconds, not after a read timeout.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.down = False
        self.served: Optional[list[str]] = None
        self.checked_at = float("-inf")
        self._recovery: Optional[asyncio.Task] = None

    def stale(self) -> bool:
        return time.monotonic() - self.checked_at > _FRESH_S

    def mark_ok(self) -> None:
        self.down = False
        self.checked_at = time.monotonic()

    def mark_down(self) -> None:
        self.down = True
        self.served = None
        self.checked_at = time.monotonic()
        if self._recovery is None or self._recovery.done():
            try:
                self._recovery = asyncio.get_running_loop().create_task(self._recover())
            except RuntimeError:
                self._recovery = None

    async def _recover(self) -> None:
        while self.down:
            await asyncio.sleep(_BREAKER_S)
            await self.probe()

    async def probe(self, timeout: float = _PROBE_TIMEOUT_S) -> Optional[list[str]]:
        cfg = _local_vllm_config("execution")
        if cfg is None:
            return None
        served = await _probe_local(cfg["base_url"], cfg["api_key"], timeout)
        if served is None:
            if not self.down:
                logger.warning("Local LLM at %s is unreachable; using the cloud cascade.", cfg["base_url"])
            self.mark_down()
        else:
            if self.down:
                logger.info("Local LLM at %s is back.", cfg["base_url"])
            self.served = served
            self.mark_ok()
        return served

    async def usable(self, model: str) -> bool:
        """Whether a chat call should try the local server for ``model`` now."""
        if self.down:
            return False
        if self.stale():
            await self.probe()
        return self.served is not None and model in self.served


local_health = LocalHealth()


class FallbackChatCompletions:
    def __init__(self, cascades: list[dict[str, str]]):
        self.clients: list[dict[str, Any]] = []
        for c in cascades:
            provider = c["provider"]
            if provider == "local_vllm":
                base_url = os.getenv("LOCAL_LLM_BASE_URL", "").strip()
                if not base_url:
                    continue
                api_key = os.getenv("LOCAL_LLM_API_KEY", "EMPTY")
                client = AsyncOpenAI(
                    api_key=api_key, base_url=base_url, timeout=LOCAL_TIMEOUT, max_retries=0
                )
                self.clients.append(
                    {"client": client, "model": c["model"], "provider": "local_vllm"}
                )
                continue

            provider_cfg = PROVIDERS.get(provider)
            if not provider_cfg:
                continue
            api_key = os.getenv(provider_cfg["api_key_env"])
            if not api_key:
                logger.warning(
                    f"Skipping {provider} due to missing {provider_cfg['api_key_env']}"
                )
                continue
            client = AsyncOpenAI(
                api_key=api_key, base_url=provider_cfg["base_url"], timeout=CLOUD_TIMEOUT
            )
            self.clients.append(
                {"client": client, "model": c["model"], "provider": provider}
            )

    async def create(self, *, think: bool = False, **kwargs: Any) -> Any:
        """OpenAI ``chat.completions.create`` with fallback across providers.

        ``think`` turns on the local model's thinking mode when it has one;
        cloud providers ignore it.
        """
        if not self.clients:
            raise RuntimeError(
                "No configured LLM providers available. "
                "Set cloud API keys or configure LOCAL_LLM_BASE_URL (see hpc/README.md)."
            )

        last_exception = None
        for index, config in enumerate(self.clients):
            client: AsyncOpenAI = config["client"]
            model: str = config["model"]
            is_local = config["provider"] == "local_vllm"
            has_fallback = index < len(self.clients) - 1

            if is_local and has_fallback and not await local_health.usable(model):
                continue

            if is_local:
                call_kwargs = _local_call_kwargs(kwargs, think)
            else:
                call_kwargs = _strip_reasoning(kwargs)
            call_kwargs = {**call_kwargs, "model": model}

            try:
                response = await client.chat.completions.create(**call_kwargs)
                if is_local:
                    local_health.mark_ok()
                return response
            except (RateLimitError, APIError, InternalServerError) as e:
                if is_local and _is_local_outage(e):
                    local_health.mark_down()
                logger.warning(
                    f"Provider {config['provider']} ({model}) failed: {e}. Falling back..."
                )
                last_exception = e
                continue
            except Exception as e:
                if is_local and isinstance(e, httpx.HTTPError):
                    local_health.mark_down()
                logger.warning(
                    f"Provider {config['provider']} ({model}) unexpected error: {e}. Falling back..."
                )
                last_exception = e
                continue

        if last_exception:
            raise last_exception
        raise RuntimeError("No LLM providers succeeded.")


class FallbackChat:
    def __init__(self, cascades: list[dict[str, str]]):
        self.completions = FallbackChatCompletions(cascades)


class FallbackLLMClient:
    def __init__(self, cascades: list[dict[str, str]]):
        self.chat = FallbackChat(cascades)


def get_llm_client(agent_type: str) -> FallbackLLMClient:
    """Return a FallbackLLMClient configured for the specific agent type."""
    cascades = _build_cascades(agent_type)
    return FallbackLLMClient(cascades)


async def _probe_local(base_url: str, api_key: str, timeout: float) -> list[str] | None:
    """Return the model ids served at ``base_url``, or None if it is unreachable."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(
                base_url.rstrip("/") + "/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
            return [m.get("id") for m in resp.json().get("data", []) if m.get("id")]
    except (httpx.HTTPError, ValueError):
        return None


async def llm_status(timeout: float = 2.0) -> dict[str, Any]:
    """Report which backend chat will actually use right now.

    ``active`` is ``local`` when the vLLM server answers and serves the
    configured model, ``cloud`` when the cascade will fall through to cloud
    providers, and ``none`` when no call can succeed.
    """
    mode = _llm_mode()
    cloud = [p for p, cfg in PROVIDERS.items() if os.getenv(cfg["api_key_env"])]

    local: dict[str, Any] | None = None
    cfg = _local_vllm_config("execution")
    if cfg and mode in ("local", "local_first"):
        # While the breaker is open the background probe owns the answer, so a
        # dead tunnel costs the status chip nothing.
        if local_health.down or not local_health.stale():
            served = local_health.served
        else:
            served = await local_health.probe(timeout)
        local = {
            "model": cfg["model"],
            "base_url": cfg["base_url"],
            "online": served is not None,
            "serves_model": bool(served) and cfg["model"] in served,
            "served": served or [],
        }

    if local and local["serves_model"]:
        active = "local"
    elif mode == "local" and local:
        # Cascade is local-only here, so an offline server means chat fails.
        active = "none"
    elif cloud:
        active = "cloud"
    else:
        active = "none"

    return {
        "mode": mode,
        "active": active,
        "local": local,
        "cloud_providers": cloud,
        "thinking": active == "local" and local_supports_thinking(),
    }
