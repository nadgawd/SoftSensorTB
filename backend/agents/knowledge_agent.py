"""Knowledge / RAG agent — conceptual soft-sensor explanations."""

from __future__ import annotations

from typing import Any, AsyncGenerator, Iterable, Optional

from backend.agents.context import (
    history_messages,
    load_dataset_context,
    render_dataset_context,
    render_tool_history,
)
from backend.agents.llm_client import get_llm_client

_BASE_SYSTEM_PROMPT = """You are a senior chemical engineering and soft-sensor expert embedded \
in the Soft Sensor Toolbox workbench.
Provide clear, high-reasoning conceptual explanations about process data analytics,
EDA, PLS/OLS soft sensors, parity plots, preprocessing, and related ML concepts.

Rules:
- Answer questions and define terms; do not invent executable code for data mutation.
- Prefer precise engineering language with brief intuition when helpful.
- If the user should instead run a data/ML action in the toolbox, say so briefly.
- Format your answer with markdown (headers, bold, lists) for readability.
"""


def _system_prompt(
    step_hint: Optional[str] = None,
    ui_context: Optional[str] = None,
    dataset_context: Optional[str] = None,
) -> str:
    prompt = _BASE_SYSTEM_PROMPT
    if dataset_context:
        prompt += (
            "\n\nThe user's loaded dataset, for grounding examples (only mention "
            f"columns that appear here):\n{dataset_context}\n"
        )
    if step_hint:
        prompt += (
            f"\nThe user is currently on pipeline step: **{step_hint}**. "
            "Tailor your answer to be most relevant to this step."
        )
    if ui_context:
        prompt += (
            f"\n\nThe user is currently looking at this UI context (e.g. plot or metrics):\n"
            f"{ui_context}\n"
            "If their question asks about 'the plot', 'this graph', or 'the metrics', refer to this context to answer."
        )
    return prompt


async def run_knowledge_agent(
    user_input: str,
    *,
    active_dataset_version_id: Optional[str] = None,
    step_hint: Optional[str] = None,
    ui_context: Optional[str] = None,
    history: Optional[Iterable[Any]] = None,
    think: bool = False,
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Answer conceptual questions with the knowledge LLM cascade.

    Streams ``reasoning`` (thinking mode only) and ``token`` events, then a
    ``final_state`` with ``ui_update_required=False``.
    """
    client = get_llm_client(agent_type="knowledge")
    ctx = await load_dataset_context(active_dataset_version_id)
    history = list(history or [])
    tool_history = render_tool_history(history)

    yield {"event": "status", "data": "Writing an explanation"}
    response = await client.chat.completions.create(
        temperature=0.3,
        max_tokens=2048,
        messages=[
            {
                "role": "system",
                "content": _system_prompt(
                    step_hint, ui_context, render_dataset_context(ctx) if ctx else None
                ) + (f"\n\n{tool_history}" if tool_history else ""),
            },
            *history_messages(history),
            {"role": "user", "content": user_input},
        ],
        stream=True,
        think=think,
    )

    async for chunk in response:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        reasoning = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
        if reasoning:
            yield {"event": "reasoning", "data": reasoning}
        if delta.content:
            yield {"event": "token", "data": delta.content}

    yield {
        "event": "final_state",
        "data": {
            "ui_update_required": False,
            "active_dataset_version_id": active_dataset_version_id,
            "plot_data": None,
            "table_preview": None,
            "model_metrics": None,
            "tool_calls_made": [],
        }
    }
