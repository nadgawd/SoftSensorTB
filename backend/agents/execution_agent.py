"""Execution agent — tool-calling loop over MCP EDA / modeling tools."""

from __future__ import annotations

import inspect
import json
import re
import time
from typing import Any, AsyncGenerator, Iterable, Optional

from backend.agents.context import (
    DatasetContext,
    history_messages,
    load_dataset_context,
    parse_ui_state,
    render_dataset_context,
    render_tool_history,
    render_ui_state,
)
from backend.agents.llm_client import get_llm_client
from backend.agents.ui_tools import SET_MODEL_FEATURES, UI_TOOL_SCHEMAS, apply_feature_selection
from backend.mcp_server import LLM_TOOL_SCHEMAS, TOOL_FUNCTIONS

MAX_TOOL_ROUNDS = 20

# An answer that reports work as done. When no tool ran this turn, such a claim
# is invented — typically the model copying the shape of an earlier reply in
# the history ("Parity Plot Generated Successfully") for a plot it never made.
_ACTION_CLAIM = re.compile(
    r"\b(generated|plotted|displayed|trained|created|applied|updated|"
    r"now (?:shown|visible|available)|has been (?:shown|added|removed))\b",
    re.IGNORECASE,
)
_NO_TOOL_NUDGE = (
    "[Automatic check] Your last reply called no tool, so nothing was generated, "
    "plotted, trained or changed, and the user sees nothing new. If the user asked "
    "for an action, call the right tool now. If they only asked something the "
    "context already answers, answer again without claiming any action."
)
_ALL_TOOL_SCHEMAS = [*LLM_TOOL_SCHEMAS, *UI_TOOL_SCHEMAS]
_MAX_ARGS_CHARS = 400
_MAX_SUMMARY_CHARS = 240

_MUTATING_TOOLS = frozenset(
    {
        "remove_missing_data",
        "remove_outliers",
        "normalize_data",
        "transform_features",
        "encode_categorical",
        "reduce_dimensions",
        "feature_engineer",
        "balance_data",
        "drop_columns",
        "execute_formula",
        "rename_columns",
        # Time-series
        "rolling_aggregate",
        "create_lag_features",
        # Advanced feature selection & data quality
        "select_features_rfe",
        "detect_anomalies",
        "drop_collinear_features",
    }
)
_PLOT_TOOLS = frozenset({
    "generate_custom_plot",
    "generate_parity_plot",
    "generate_residuals_plot",
    "generate_importance_plot",
    "generate_coefficients_plot",
})
_TABLE_TOOLS = frozenset({"run_duckdb_query", "get_column_statistics"})
_METRIC_TOOLS = frozenset({"train_soft_sensor"})


def _execution_system_prompt(
    ctx: Optional[DatasetContext],
    ui: dict[str, Any],
    step_hint: Optional[str] = None,
    tool_history: str = "",
) -> str:
    step_ctx = f"\nCURRENT PIPELINE STEP: {step_hint}" if step_hint else ""
    if tool_history:
        step_ctx += f"\n{tool_history}"
    return f"""You are the Soft Sensor Toolbox execution agent.
You solve data exploration, preprocessing, plotting, feature selection and
soft-sensor modeling tasks by calling the provided tools. Never invent Python
code or mutate data yourself. Answer questions about the user's data by
computing them with run_duckdb_query, get_column_statistics or the plot tools.

{render_dataset_context(ctx)}

{render_ui_state(ui)}{step_ctx}

Rules:
- Use ONLY column names listed under COLUMNS, spelled exactly. Never use
  placeholder names such as sensor1_value, temperature, pressure or A/B/C.
  Resolve "all numeric columns", "these columns", "the same ones" from COLUMNS,
  the UI STATE and the earlier conversation.
- Do not ask the user for anything you can look up. Columns are listed above,
  plot tools find the newest trained model on their own (omit model_id), and
  list_trained_models lists every model. Ask a clarifying question only when
  the request is genuinely ambiguous.
- Training: when the user does not name the target or features, use the UI
  STATE target and selected features; if none are selected, use every numeric
  column except the target and any Predicted_* column.
- To change the target or the selected features in the Feature Selection step,
  call set_model_features (it accepts wildcards such as '*_lag*'). For "only
  these" / "select the new ones and unselect the old ones", pass the complete
  new list in `select` in a single call.
- Never tell the user to call a tool or pass parameters; call it yourself.
- The dataset version id is injected into tools automatically.
- If a tool returns an error, correct the arguments and retry. Keep failed
  attempts out of the final answer unless they matter to the user.
- The final answer is a concise engineering summary of what was done and the
  key results (column names, row counts, metric values). Do not claim UI
  updates that the tools did not produce.
"""


def _plot_title(result: Any) -> Optional[str]:
    if not isinstance(result, dict):
        return None
    title = (result.get("layout") or {}).get("title")
    return title.get("text") if isinstance(title, dict) else title


def _serialize_tool_result(tool_name: str, result: Any) -> str:
    if tool_name in _PLOT_TOOLS:
        return json.dumps({
            "status": "success",
            "message": "Plotly figure generated and shown in the UI.",
            "title": _plot_title(result),
        })

    if isinstance(result, tuple) and len(result) == 2:
        return json.dumps(
            {
                "summary_message": result[1],
                "new_version_id": result[0],
                "note": "A new dataset version was created from the previous one and is now active; later tools use it automatically.",
            },
            default=str,
        )
    return json.dumps(result, default=str)


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _display_args(arguments: dict[str, Any]) -> str:
    shown = {k: v for k, v in arguments.items() if k != "dataset_version_id"}
    return _truncate(json.dumps(shown, default=str), _MAX_ARGS_CHARS) if shown else ""


def _summarize_tool_result(tool_name: str, result: Any) -> str:
    """One-line human summary shown in the chat activity log."""
    if tool_name in _PLOT_TOOLS:
        return _plot_title(result) or "Figure generated"
    if isinstance(result, tuple) and len(result) == 2:
        return _truncate(f"{result[1]} → version {str(result[0])[:8]}", _MAX_SUMMARY_CHARS)
    if isinstance(result, dict):
        if tool_name == "train_soft_sensor":
            return (
                f"{result.get('algorithm')} · R² {result.get('r2_score', 0):.3f} · "
                f"RMSE {result.get('rmse', 0):.3g} · model {str(result.get('model_id'))[:8]}"
            )
        if tool_name == SET_MODEL_FEATURES:
            feats = result.get("selected_features") or []
            return _truncate(f"target {result.get('target_column') or '—'} · {len(feats)} features: {', '.join(feats)}", _MAX_SUMMARY_CHARS)
        if tool_name == "list_trained_models":
            return f"{result.get('count', 0)} model(s) found"
        if isinstance(result.get("rows"), list):
            return f"{len(result['rows'])} row(s) returned"
    return _truncate(json.dumps(result, default=str), _MAX_SUMMARY_CHARS)


def _inject_dataset_version(
    tool_name: str,
    arguments: dict[str, Any],
    active_dataset_version_id: str,
) -> dict[str, Any]:
    """Pin tools to the active dataset version, whatever id the LLM supplied.

    The request was authorised for the active version only, so a model-chosen
    id (hallucinated or prompt-injected) must never reach another dataset.
    """
    args = {k: v for k, v in arguments.items() if k != "session"}
    fn = TOOL_FUNCTIONS.get(tool_name)
    if fn is None:
        return args
    params = inspect.signature(fn).parameters
    if "dataset_version_id" in params:
        args["dataset_version_id"] = active_dataset_version_id
    return args


def _update_ui_state(
    state: dict[str, Any],
    *,
    tool_name: str,
    result: Any,
) -> None:
    """Fold tool outputs into the rich UI payload accumulators."""
    if tool_name in _MUTATING_TOOLS and isinstance(result, tuple) and len(result) == 2:
        new_version_id, _summary = result
        state["active_dataset_version_id"] = str(new_version_id)
        state["ui_update_required"] = True
        return

    if tool_name in _PLOT_TOOLS and isinstance(result, dict):
        state["plot_data"] = result
        state["ui_update_required"] = True
        return

    if tool_name in _TABLE_TOOLS and isinstance(result, dict):
        rows = result.get("rows") or []
        # Cap preview size for the chat response payload.
        state["table_preview"] = rows[:50]
        state["ui_update_required"] = True
        return

    if tool_name == SET_MODEL_FEATURES and isinstance(result, dict):
        state["selected_features"] = result["selected_features"]
        state["target_variable"] = result["target_column"]
        state["ui_update_required"] = True
        return

    if tool_name in _METRIC_TOOLS and isinstance(result, dict):
        state["model_metrics"] = {
            "model_id": result.get("model_id"),
            "algorithm": result.get("algorithm"),
            "r2_score": result.get("r2_score"),
            "rmse": result.get("rmse"),
            "feature_importances": result.get("feature_importances")
            or result.get("coefficients"),
            "n_train": result.get("n_train"),
            "n_test": result.get("n_test"),
            "target_column": result.get("target_column"),
            "feature_columns": result.get("feature_columns"),
        }
        if "new_version_id" in result:
            state["active_dataset_version_id"] = result["new_version_id"]
        state["ui_update_required"] = True


async def _dispatch_tool(
    tool_name: str,
    arguments: dict[str, Any],
    active_dataset_version_id: str,
) -> Any:
    if tool_name not in TOOL_FUNCTIONS:
        raise ValueError(f"Unknown tool: {tool_name!r}")
    args = _inject_dataset_version(tool_name, arguments, active_dataset_version_id)
    fn = TOOL_FUNCTIONS[tool_name]
    result = fn(**args)
    if inspect.isawaitable(result):
        result = await result
    return result


def _run_ui_tool(
    tool_name: str,
    arguments: dict[str, Any],
    ctx: Optional[DatasetContext],
    ui_state: dict[str, Any],
) -> dict[str, Any]:
    if tool_name != SET_MODEL_FEATURES:
        raise ValueError(f"Unknown tool: {tool_name!r}")
    if ctx is None:
        raise ValueError("The dataset schema could not be loaded, so features cannot be selected.")
    return apply_feature_selection(
        columns=ctx.column_names,
        numeric_columns=ctx.numeric_columns,
        current_features=ui_state["selected_features"] or [],
        current_target=ui_state["target_variable"],
        target_column=arguments.get("target_column"),
        select=arguments.get("select"),
        add=arguments.get("add"),
        remove=arguments.get("remove"),
    )


def _reasoning_delta(delta: Any) -> Optional[str]:
    # vLLM >= 0.13 names the field `reasoning`; older servers use `reasoning_content`.
    return getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)


async def _stream_completion(
    client: Any,
    messages: list[dict[str, Any]],
    max_tokens: int,
    sink: dict[str, Any],
    think: bool = False,
) -> AsyncGenerator[tuple[str, str], None]:
    """Stream one model turn.

    Yields ``("reasoning", text)`` and ``("token", text)`` deltas and collects
    content, reasoning and tool calls in ``sink``.
    """
    stream = await client.chat.completions.create(
        temperature=0.1,
        max_tokens=max_tokens,
        tools=_ALL_TOOL_SCHEMAS,
        tool_choice="auto",
        messages=messages,
        stream=True,
        think=think,
    )
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        reasoning = _reasoning_delta(delta)
        if reasoning:
            sink["reasoning"] += reasoning
            yield "reasoning", reasoning
        if delta.content:
            sink["content"] += delta.content
            yield "token", delta.content
        for tc in delta.tool_calls or []:
            call = sink["tool_calls"].setdefault(
                tc.index,
                {"id": tc.id, "type": "function", "function": {"name": "", "arguments": ""}},
            )
            if tc.id and not call["id"]:
                call["id"] = tc.id
            if tc.function and tc.function.name:
                call["function"]["name"] += tc.function.name
            if tc.function and tc.function.arguments:
                call["function"]["arguments"] += tc.function.arguments


def _is_recoverable(err: Exception) -> bool:
    text = str(err).lower()
    return any(m in text for m in (
        "context_length", "reduce the length", "internal server error",
        "upstream error", "500", "502", "503", "504", "529",
    ))


async def run_execution_agent(
    user_input: str,
    dataset_version_id: str,
    step_hint: Optional[str] = None,
    ui_context: Optional[str] = None,
    history: Optional[Iterable[Any]] = None,
    think: bool = False,
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Run the tool-calling execution loop, streaming progress for the chat UI.

    Events: ``status`` (phase text), ``reasoning`` (thinking-mode text),
    ``token`` (answer text), ``thought`` (text the model wrote before calling
    tools — the frontend moves the streamed draft into the activity log),
    ``retract`` (drop the streamed draft: it claimed work no tool did),
    ``tool_start`` / ``tool_end`` and a closing ``final_state`` with the UI
    payload.
    """
    client = get_llm_client(agent_type="execution")
    ui = parse_ui_state(ui_context)
    ui_state: dict[str, Any] = {
        "ui_update_required": False,
        "active_dataset_version_id": dataset_version_id,
        "plot_data": None,
        "table_preview": None,
        "model_metrics": None,
        "selected_features": ui.get("selectedFeatures") or [],
        "target_variable": ui.get("targetVariable"),
    }
    features_changed = False
    tool_calls_made: list[str] = []

    yield {"event": "status", "data": "Reading dataset schema and trained models"}
    ctx = await load_dataset_context(dataset_version_id)

    history = list(history or [])
    tool_history = render_tool_history(history)
    user_msg = {"role": "user", "content": user_input}
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": ""},
        *history_messages(history),
        user_msg,
    ]

    final_message = ""
    previous_tool_calls_str = None
    claim_checked = False

    for round_no in range(MAX_TOOL_ROUNDS):
        if ctx is None or ctx.version_id != ui_state["active_dataset_version_id"]:
            ctx = await load_dataset_context(ui_state["active_dataset_version_id"])
        messages[0] = {
            "role": "system",
            "content": _execution_system_prompt(
                ctx, {**ui, "selectedFeatures": ui_state["selected_features"],
                      "targetVariable": ui_state["target_variable"]}, step_hint, tool_history,
            ),
        }
        yield {"event": "status", "data": "Reviewing tool results" if tool_calls_made else "Planning"}

        sink: dict[str, Any] = {"content": "", "reasoning": "", "tool_calls": {}}
        try:
            async for kind, text in _stream_completion(client, messages, 4096, sink, think):
                yield {"event": kind, "data": text}
        except Exception as api_err:  # noqa: BLE001
            if not _is_recoverable(api_err):
                final_message = (
                    f"I encountered an API error: {str(api_err)[:300]}. "
                    "Please try rephrasing your request."
                )
                yield {"event": "token", "data": final_message}
                break
            # Context overflow or upstream crash: drop history, keep the current turn, retry once.
            yield {"event": "status", "data": "Model call failed — retrying with a shorter context"}
            messages = [messages[0], user_msg]
            sink = {"content": "", "reasoning": "", "tool_calls": {}}
            try:
                async for kind, text in _stream_completion(client, messages, 2048, sink, think):
                    yield {"event": kind, "data": text}
            except Exception as retry_err:  # noqa: BLE001
                final_message = (
                    "The conversation grew too large and I couldn't recover. "
                    f"Please start a new question. ({retry_err})"
                )
                yield {"event": "token", "data": final_message}
                break

        current_content = sink["content"]
        tool_calls = list(sink["tool_calls"].values())

        # Break out of infinite tool loops if the exact same tools and args are called sequentially
        current_tool_calls_str = json.dumps([
            {"name": tc["function"]["name"], "args": tc["function"]["arguments"]}
            for tc in tool_calls
        ], sort_keys=True)

        if tool_calls and current_tool_calls_str == previous_tool_calls_str:
            yield {"event": "thought", "data": current_content.strip()}
            final_message = "*I stopped because I was repeating the same failing action. The tool errors are in the activity log above.*"
            yield {"event": "token", "data": final_message}
            break

        previous_tool_calls_str = current_tool_calls_str

        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": current_content,
        }
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        if sink["reasoning"].strip():
            # Qwen3.5's template replays reasoning from earlier rounds of the current turn.
            assistant_msg["reasoning_content"] = sink["reasoning"].strip()
        messages.append(assistant_msg)

        if not tool_calls:
            if not tool_calls_made and not claim_checked and _ACTION_CLAIM.search(current_content):
                claim_checked = True
                yield {"event": "retract", "data": ""}
                messages.append({"role": "user", "content": _NO_TOOL_NUDGE})
                continue
            final_message = current_content.strip()
            break

        yield {"event": "thought", "data": current_content.strip()}

        for i, tc in enumerate(tool_calls):
            name = tc["function"]["name"]
            call_id = tc["id"] or f"call_{round_no}_{i}"
            tool_calls_made.append(name)
            started = time.perf_counter()
            raw_args = tc["function"]["arguments"] or "{}"
            try:
                arguments = json.loads(raw_args) if raw_args.strip() else {}
            except ValueError:
                arguments = None
            yield {"event": "tool_start", "data": {
                "id": call_id,
                "name": name,
                "args": _display_args(arguments) if isinstance(arguments, dict) else _truncate(raw_args, _MAX_ARGS_CHARS),
            }}
            try:
                if not isinstance(arguments, dict):
                    raise ValueError("Tool arguments must be a JSON object")
                if name == SET_MODEL_FEATURES:
                    result = _run_ui_tool(name, arguments, ctx, ui_state)
                    features_changed = True
                else:
                    result = await _dispatch_tool(name, arguments, ui_state["active_dataset_version_id"])
                _update_ui_state(ui_state, tool_name=name, result=result)
                payload = _serialize_tool_result(name, result)
                end = {"ok": True, "summary": _summarize_tool_result(name, result)}
            except Exception as exc:  # noqa: BLE001 — surface tool errors to the LLM
                payload = json.dumps({"error": str(exc)})
                end = {"ok": False, "summary": _truncate(str(exc), _MAX_SUMMARY_CHARS)}
            ms = int((time.perf_counter() - started) * 1000)
            yield {"event": "tool_end", "data": {"id": call_id, "name": name, "ms": ms, **end}}

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": payload,
                }
            )
    else:
        final_message = (
            final_message
            or "Reached the maximum number of tool rounds. Partial results may be available."
        )
        yield {"event": "token", "data": final_message}

    if not final_message:
        final_message = "Completed tool execution."
        yield {"event": "token", "data": final_message}

    yield {
        "event": "final_state",
        "data": {
            "ui_update_required": bool(ui_state["ui_update_required"]),
            "active_dataset_version_id": ui_state["active_dataset_version_id"],
            "plot_data": ui_state["plot_data"],
            "table_preview": ui_state["table_preview"],
            "model_metrics": ui_state["model_metrics"],
            "selected_features": ui_state["selected_features"] if features_changed else None,
            "target_variable": ui_state["target_variable"] if features_changed else None,
            "tool_calls_made": tool_calls_made,
        }
    }
