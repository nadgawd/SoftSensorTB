"""Knowledge / RAG agent — conceptual soft-sensor explanations."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, AsyncGenerator, Iterable, Optional

from backend.agents.context import (
    history_messages,
    load_dataset_context,
    parse_ui_state,
    render_dataset_context,
    render_manual_actions,
    render_tool_history,
)
from backend.agents.llm_client import get_llm_client
from backend.agents.router import SCOPE_RULE

logger = logging.getLogger(__name__)

_EMPTY_ANSWER = "*The model returned an empty answer. Please ask again.*"


def _failure_message(err: Exception, partial: bool) -> str:
    if partial:
        return "\n\n*The answer was cut off because the model connection dropped. Please ask again.*"
    return (
        "*I couldn't reach any AI model just now (all providers failed or are busy). "
        f"Please try again in a minute.*\n\n`{type(err).__name__}: {str(err)[:200]}`"
    )

_BASE_SYSTEM_PROMPT = f"""You are a senior chemical engineering and soft-sensor expert embedded \
in the Soft Sensor Toolbox workbench.
Provide clear, high-reasoning conceptual explanations about process data analytics,
EDA, PLS/OLS soft sensors, parity plots, preprocessing, and related ML concepts.

Rules:
- Answer questions and define terms; do not invent executable code for data mutation.
- Prefer precise engineering language with brief intuition when helpful.
- If the user should instead run a data/ML action in the toolbox, say so briefly.
- Format your answer with markdown (headers, bold, lists) for readability.
- {SCOPE_RULE}
"""


DEPLOYMENT_NOTES = """How this toolbox deploys a model (the Deployment step):
- Deploying compiles every step from the upload to the training data into steps that run on live rows. \
Fitted steps (scaling, filling, encoding, PCA) reuse the values learnt on the training data. Window steps \
(lags, rolling, EWM) keep a buffer of recent rows. Every step is checked to reproduce the stored version \
exactly, and the whole chain is checked on the training rows, on the live columns only, and row by row as a stream.
- A plant connector (a small Python script run next to the plant) reads the inputs from a CSV export, \
OPC UA, MQTT or the PI Web API every cycle. It sends them over HTTPS with the deployment's API key \
(outbound only) and can write the prediction back to a tag.
- Each prediction has a status. It is ok, or warning when an input is outside the training range, a sensor \
is stuck, or the plant is outside the training operating region (Hotelling T² or SPE above its 99% training \
limit). It is suppressed when a training filter removes the row (such as a shutdown), when an input is \
missing, or when there is not enough history yet. Repeated or older rows are ignored.
- Lab results (with their sample time) are matched to the prediction at that time. They update a bias: \
bias ← (1 − g)·bias + g·(lab − prediction), and the corrected value is prediction + bias. If the recent \
lab error exceeds twice the test RMSE, the model should be retrained.
- The Simulate panel replays the deployment's own data as a live feed to try all of this before connecting a plant."""

_DEPLOYMENT_TOPIC = re.compile(
    r"\b(?:deploy\w*|live|real[\s-]?time|connector|opc|mqtt|historian|pi\s+web|api\s+key|t2|t²|spe|"
    r"hotelling|bias|drift|suppress\w*|simulat\w*)\b",
    re.IGNORECASE,
)


def _system_prompt(
    step_hint: Optional[str] = None,
    ui_context: Optional[str] = None,
    dataset_context: Optional[str] = None,
    user_input: Optional[str] = None,
) -> str:
    prompt = _BASE_SYSTEM_PROMPT
    if (step_hint and "deploy" in step_hint.lower()) or (user_input and _DEPLOYMENT_TOPIC.search(user_input)):
        prompt += "\n\n" + DEPLOYMENT_NOTES
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
        ui = parse_ui_state(ui_context)
        shown = ui_context
        if ui:
            manual = render_manual_actions(ui.pop("manualActions", None))
            shown = json.dumps(ui, default=str, ensure_ascii=False) + (f"\n{manual}" if manual else "")
        prompt += (
            f"\n\nThe user is currently looking at this UI context (e.g. plot or metrics):\n"
            f"{shown}\n"
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
    answered = False
    try:
        response = await client.chat.completions.create(
            temperature=0.3,
            max_tokens=2048,
            messages=[
                {
                    "role": "system",
                    "content": _system_prompt(
                        step_hint, ui_context, render_dataset_context(ctx) if ctx else None, user_input,
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
                answered = True
                yield {"event": "token", "data": delta.content}
    except Exception as err:  # noqa: BLE001
        logger.exception("Knowledge agent failed")
        yield {"event": "token", "data": _failure_message(err, answered)}
    else:
        if not answered:
            yield {"event": "token", "data": _EMPTY_ANSWER}

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
