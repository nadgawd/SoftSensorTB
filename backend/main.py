"""Multi-agent Soft Sensor Toolbox orchestration API."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, Literal, Optional

import json

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from backend.agents.execution_agent import run_execution_agent
from backend.agents.knowledge_agent import run_knowledge_agent
from backend.agents.router import classify_intent
from backend.api.datasets import router as datasets_router
from backend.api.project import router as project_router
from backend.auth import User, auth_required, get_current_user
from backend.database import AsyncSessionLocal, init_db
from backend.limits import chat_limit_per_hour, chat_limiter
from backend.mcp_server.eda_tools import _validate_uuid_str
from backend.services.projects import owned_version


@asynccontextmanager
async def _lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="Soft Sensor Toolbox API",
    description="Multi-agent orchestration backend for EDA, lineage, and soft sensors.",
    version="0.1.0",
    lifespan=_lifespan,
)

# The frontend authenticates with an `Authorization: Bearer` header, not cookies,
# so credentials stay off. Deployments set CORS_ALLOW_ORIGINS to the Pages origin;
# the wildcard default is for local development only.
_cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
    expose_headers=["Retry-After"],
)

app.include_router(datasets_router)
app.include_router(project_router)


def _allowed_origin(request: Request) -> Optional[str]:
    if "*" in _cors_origins:
        return "*"
    origin = request.headers.get("origin")
    return origin if origin in _cors_origins else None


@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all for unhandled exceptions — returns JSON so the frontend
    always receives a parseable error body instead of a connection drop.
    """
    # Unhandled errors bypass CORSMiddleware, so the header is added here.
    origin = _allowed_origin(request)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error." if auth_required() else str(exc)[:500]},
        headers={"Access-Control-Allow-Origin": origin} if origin else None,
    )


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(default="", max_length=20_000)
    tools: list[str] = Field(default_factory=list, description="Tools the assistant ran that turn.")
    intent: Optional[str] = None


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8_000, description="Natural-language user turn.")
    history: list[ChatTurn] = Field(
        default_factory=list,
        max_length=50,
        description="Earlier turns of this conversation, oldest first.",
    )
    dataset_version_id: Optional[str] = Field(
        default=None,
        description="Active dataset version UUID (required for EXECUTE intents).",
    )
    step_hint: Optional[str] = Field(
        default=None,
        description="Current pipeline step name for context-aware agent responses.",
    )
    ui_context: Optional[str] = Field(
        default=None,
        max_length=50_000,
        description="Stringified JSON of current UI state (e.g. visible plot/metrics) for the RAG agent.",
    )
    think: bool = Field(
        default=False,
        description="Use the local model's thinking mode, if it has one.",
    )

    @field_validator("dataset_version_id")
    @classmethod
    def _optional_uuid(cls, value: Optional[str]) -> Optional[str]:
        if value is None or value == "":
            return None
        return _validate_uuid_str(value)


class ChatResponse(BaseModel):
    message: str
    ui_update_required: bool
    active_dataset_version_id: Optional[str] = None
    plot_data: Optional[dict[str, Any]] = None
    table_preview: Optional[list[dict[str, Any]]] = None
    model_metrics: Optional[dict[str, Any]] = None
    intent: Optional[str] = Field(
        default=None,
        description="Router decision (EXECUTE or RAG) for debugging / UI badges.",
    )
    tool_calls_made: list[str] = Field(
        default_factory=list,
        description="Ordered list of tool names called during execution.",
    )


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
async def root() -> dict[str, str]:
    return {"status": "ok"}


# HEAD is accepted because uptime monitors (e.g. UptimeRobot) probe with it by default.
@app.api_route("/health", methods=["GET", "HEAD"])
async def health(deep: bool = False) -> dict[str, str]:
    """``?deep=true`` also checks the database and file storage (for deploy checks)."""
    if not deep:
        return {"status": "ok"}
    from sqlalchemy import text
    from starlette.concurrency import run_in_threadpool

    from backend.storage import get_storage

    checks: dict[str, str] = {}
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:  # noqa: BLE001 — report, don't raise
        checks["database"] = f"error: {type(exc).__name__}: {str(exc)[:200]}"
    try:
        await run_in_threadpool(get_storage().ping)
        checks["storage"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["storage"] = f"error: {type(exc).__name__}: {str(exc)[:200]}"
    checks["status"] = "ok" if all(v == "ok" for v in checks.values()) else "degraded"
    return checks


@app.get("/api/llm/status")
async def get_llm_status() -> dict[str, Any]:
    from backend.agents.llm_client import llm_status
    return await llm_status()


@app.get("/api/tools")
async def get_tools() -> list[dict[str, Any]]:
    from backend.mcp_server import LLM_TOOL_SCHEMAS
    return LLM_TOOL_SCHEMAS


@app.post("/chat")
async def chat(request: ChatRequest, user: User = Depends(get_current_user)) -> StreamingResponse:
    """
    Route the user turn, run the matching agent, and stream a rich UI payload.
    """
    chat_limiter.check(user.id, chat_limit_per_hour())
    if request.dataset_version_id:
        # Tools are pinned to this version, so checking it here covers every tool call.
        async with AsyncSessionLocal() as session:
            await owned_version(session, request.dataset_version_id, user)

    previous_intent = next(
        (t.intent for t in reversed(request.history) if t.role == "assistant" and t.intent),
        None,
    )
    intent = await classify_intent(
        request.message,
        has_dataset=bool(request.dataset_version_id),
        previous_intent=previous_intent,
    )

    async def event_generator():
        yield f"data: {json.dumps({'event': 'intent', 'data': intent})}\n\n"
        route = "execution agent (tools)" if intent == "EXECUTE" and request.dataset_version_id else "knowledge agent"
        yield f"data: {json.dumps({'event': 'status', 'data': f'Routed to the {route}'})}\n\n"

        if intent == "RAG" or not request.dataset_version_id:
            # No dataset → always answer via RAG with a friendly note if EXECUTE was intended
            if intent == "EXECUTE" and not request.dataset_version_id:
                prefix = (
                    "⚠ **No dataset uploaded yet** — I can't execute data operations without one.\n\n"
                    "Here's some context that may help:\n\n"
                )
                yield f"data: {json.dumps({'event': 'token', 'data': prefix})}\n\n"
                
            async for chunk in run_knowledge_agent(
                request.message,
                active_dataset_version_id=request.dataset_version_id,
                step_hint=request.step_hint,
                ui_context=request.ui_context,
                history=request.history,
                think=request.think,
            ):
                yield f"data: {json.dumps(chunk)}\n\n"
            return

        async for chunk in run_execution_agent(
            request.message,
            request.dataset_version_id,
            step_hint=request.step_hint,
            ui_context=request.ui_context,
            history=request.history,
            think=request.think,
        ):
            yield f"data: {json.dumps(chunk)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
