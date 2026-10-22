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
from backend.agents.router import SCOPE_RULE
from backend.agents.tool_selection import select_schemas
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
# The claim check only applies when the user asked for something to be done.
# Summaries and advice ("summarize the pipeline", "what are the next steps?")
# legitimately recount earlier work as "trained" / "generated".
_ACTION_REQUEST = re.compile(
    r"\b(plot|chart|graph|draw|visuali[sz]e|show|display|generate|create|make|"
    r"train|retrain|fit|build|run|apply|remove|drop|delete|fill|impute|clean|"
    r"normali[sz]e|scale|standardi[sz]e|transform|select|filter|split|detect|"
    r"add|compute|calculate|export|set|use|redo|rerun)\b",
    re.IGNORECASE,
)
# The user never sees this message, and the retried reply replaces the retracted
# one on screen, so it must read as a fresh answer rather than a reply to the check.
_NO_TOOL_NUDGE = (
    "[Automatic check — hidden from the user] Your last reply called no tool, so "
    "nothing was generated, plotted, trained or changed this turn, and that reply "
    "has been removed from the user's screen. If the user asked for an action, call "
    "the right tool now. Otherwise write your full answer to the user's message "
    "again, without claiming any new action. Do not mention this check, your "
    "previous reply, or that no tool was called."
)
_REPEAT_NOTE = (
    "This call is identical to your previous one, so it was not run again; its result "
    "is above. Use it and continue with the next step, or answer if you are done."
)
_ALL_TOOL_SCHEMAS = [*LLM_TOOL_SCHEMAS, *UI_TOOL_SCHEMAS]
_MAX_ARGS_CHARS = 400
_MAX_SUMMARY_CHARS = 240
# Every round resends the whole conversation, so tool results are what grows
# it. The user sees full tables in the app; the model gets enough to decide.
_MAX_RESULT_CHARS = 8000
_SEEN_RESULT_CHARS = 1200
_TIGHT_RESULT_CHARS = 500

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
        "resample_time_series",
        # Advanced feature selection & data quality
        "select_features_rfe",
        "select_k_best",
        "detect_anomalies",
        "drop_collinear_features",
        # Rows, types and text
        "drop_rows",
        "drop_duplicates",
        "convert_types",
        "clean_text",
        # Sorting, reshaping, combining and mapping
        "sort_data",
        "reshape_data",
        "combine_datasets",
        "mask_values",
        "map_values",
        "apply_function",
        # Model-based feature selection
        "select_features_model",
    }
)
_PLOT_TOOLS = frozenset({
    "generate_custom_plot",
    "generate_parity_plot",
    "generate_residuals_plot",
    "generate_importance_plot",
    "generate_coefficients_plot",
    "generate_roc_curve",
    "generate_pr_curve",
    "generate_learning_curve",
})
_TABLE_TOOLS = frozenset({
    "run_duckdb_query", "get_column_statistics", "inspect_data", "list_trained_models",
    "aggregate_data", "statistical_test", "list_datasets", "view_data_split",
})
_METRIC_TOOLS = frozenset({"train_soft_sensor", "evaluate_on_test_set"})
_MAX_TABLE_ROWS = 50
_TABLE_TITLES = {
    "run_duckdb_query": "Query result",
    "get_column_statistics": "Column statistics",
    "inspect_data": "Data inspection",
    "list_trained_models": "Trained models",
    "aggregate_data": "Aggregation",
    "statistical_test": "Statistical test",
    "list_datasets": "Your datasets",
    "view_data_split": "Train / validation / test split",
}


def _execution_system_prompt(
    ctx: Optional[DatasetContext],
    ui: dict[str, Any],
    step_hint: Optional[str] = None,
    tool_history: str = "",
    omitted_tools: Iterable[str] = (),
) -> str:
    step_ctx = f"\nCURRENT PIPELINE STEP: {step_hint}" if step_hint else ""
    if tool_history:
        step_ctx += f"\n{tool_history}"
    omitted = list(omitted_tools)
    if omitted:
        step_ctx += (
            "\nMORE TOOLS (not attached to save space; call any of them by name with the "
            f"arguments you expect and its full schema is attached from the next round): {', '.join(omitted)}"
        )
    return f"""You are the Soft Sensor Toolbox execution agent.
You solve data exploration, preprocessing, plotting, feature selection and
soft-sensor modeling tasks by calling the provided tools. Never invent Python
code or mutate data yourself. Answer questions about the user's data by
computing them with run_duckdb_query, get_column_statistics, inspect_data (head,
tail, info, describe, shape, value_counts, missing, correlation, profile,
nunique, distribution with skew / kurtosis / quantiles, alerts), aggregate_data
(groupby, pivot_table, crosstab), statistical_test (t-tests, ANOVA, chi-square,
normality, ADF stationarity) or the plot tools.

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
- Training and testing are separate: train_soft_sensor fits on the training
  rows and reports training scores only; evaluate_on_test_set then scores the
  held-out test rows. When the user asks how well a model performs, or to
  "train and test", call evaluate_on_test_set after training. view_data_split
  shows the train / validation / test rows.
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
- {SCOPE_RULE}
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
        return _fit_json(
            {
                "summary_message": result[1],
                "new_version_id": result[0],
                "note": "A new dataset version was created from the previous one and is now active; later tools use it automatically.",
            },
            _MAX_RESULT_CHARS,
        )
    return _fit_json(result, _MAX_RESULT_CHARS)


def _fit_json(obj: Any, limit: int) -> str:
    """JSON of ``obj`` within ``limit`` characters.

    An oversized dict keeps whole entries of its largest list or mapping
    (table rows, per-column statistics) and says how many were left out.
    """
    text = json.dumps(obj, default=str)
    if len(text) <= limit:
        return text
    if isinstance(obj, dict):
        sizes = {k: len(json.dumps(v, default=str)) for k, v in obj.items() if isinstance(v, (list, dict))}
        if sizes:
            key = max(sizes, key=sizes.get)
            items = list(obj[key].items()) if isinstance(obj[key], dict) else list(obj[key])
            rest = {k: v for k, v in obj.items() if k != key}
            budget = limit - len(json.dumps(rest, default=str)) - 300
            kept: list[Any] = []
            used = 0
            for item in items:
                used += len(json.dumps(item, default=str)) + 2
                if used > budget:
                    break
                kept.append(item)
            note = (
                f"{key} cut to the first {len(kept)} of {len(items)} entries to fit the model's context; "
                "the user sees the full result in the app. Ask for fewer columns or rows if you need the rest."
            )
            text = json.dumps({**rest, key: dict(kept) if isinstance(obj[key], dict) else kept, "truncated": note}, default=str)
            if len(text) <= limit:
                return text
    return _clip_result(text, limit)


def _clip_result(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}…[{len(text) - limit:,} more characters cut to save context]"


def _shrink_tool_results(messages: list[dict[str, Any]], start: int, end: int, limit: int) -> None:
    for message in messages[start:end]:
        if message.get("role") == "tool" and isinstance(message.get("content"), str):
            message["content"] = _clip_result(message["content"], limit)


def _shrink_seen_results(messages: list[dict[str, Any]], start: int) -> None:
    """Shorten tool results the model already read in an earlier round of this turn."""
    last_call = max(
        (i for i in range(start, len(messages)) if messages[i].get("role") == "assistant" and messages[i].get("tool_calls")),
        default=None,
    )
    if last_call is not None:
        _shrink_tool_results(messages, start, last_call, _SEEN_RESULT_CHARS)


def _tight_messages(messages: list[dict[str, Any]], user_msg: dict[str, Any]) -> list[dict[str, Any]]:
    """After a too-large error: drop earlier chat turns and shorten this turn's tool results, keeping its progress."""
    turn = messages[next(i for i, m in enumerate(messages) if m is user_msg) + 1:]
    out = [messages[0], user_msg, *({k: v for k, v in m.items() if k != "reasoning_content"} for m in turn)]
    _shrink_tool_results(out, 2, len(out), _TIGHT_RESULT_CHARS)
    return out


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
            cv = result.get("cv") or {}
            cv_text = (
                f" · CV ({cv.get('strategy')}) R² {cv['r2']:.3f}"
                if isinstance(cv.get("r2"), (int, float)) else ""
            )
            val = result.get("validation") or {}
            val_text = f" · validation R² {val['r2']:.3f}" if isinstance(val.get("r2"), (int, float)) else ""
            train = result.get("train") or {}
            train_text = (
                f"train R² {train['r2']:.3f} · RMSE {train['rmse']:.3g}"
                if isinstance(train.get("r2"), (int, float)) and isinstance(train.get("rmse"), (int, float))
                else "trained"
            )
            return (
                f"{result.get('algorithm')} · {train_text} on {result.get('n_train')} rows{val_text}{cv_text} · "
                f"{result.get('n_test')} test rows held out, not tested yet · model {str(result.get('model_id'))[:8]}"
            )
        if tool_name == "evaluate_on_test_set":
            test = result.get("test") or {}
            r2 = f"test R² {test['r2']:.3f}" if isinstance(test.get("r2"), (int, float)) else "test R² —"
            return (
                f"{result.get('algorithm')} · {r2} · RMSE {test.get('rmse') or 0:.3g} · MAE {test.get('mae') or 0:.3g} "
                f"on {test.get('n_rows')} held-out test rows · model {str(result.get('model_id'))[:8]}"
            )
        if tool_name == SET_MODEL_FEATURES:
            feats = result.get("selected_features") or []
            return _truncate(f"target {result.get('target_column') or '—'} · {len(feats)} features: {', '.join(feats)}", _MAX_SUMMARY_CHARS)
        if tool_name == "list_trained_models":
            return f"{result.get('count', 0)} model(s) found"
        if isinstance(result.get("rows"), list):
            prefix = f"{result['title']} · " if result.get("title") else ""
            return f"{prefix}{len(result['rows'])} row(s) returned"
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


def _cell(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, default=str)


def _table_rows(tool_name: str, result: dict[str, Any]) -> list[dict[str, Any]]:
    if tool_name == "get_column_statistics":
        return [
            {"column": name, **{k: _cell(v) for k, v in stats.items()}}
            for name, stats in (result.get("statistics") or {}).items()
        ]
    if tool_name == "list_trained_models":
        return [
            {
                "model_id": m.get("model_id"),
                "algorithm": m.get("algorithm"),
                "target": m.get("target_column"),
                "features": ", ".join(m.get("feature_columns") or []),
                "r2_score": m.get("r2_score"),
                "rmse": m.get("rmse"),
                "cv_r2": (m.get("cv") or {}).get("r2"),
                "validation_r2": (m.get("validation") or {}).get("r2"),
                "settings": _cell(m.get("hyperparameters") or None),
                "trained_at": m.get("trained_at"),
            }
            for m in result.get("models") or []
        ]
    return list(result.get("rows") or [])


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
        rows = _table_rows(tool_name, result)
        # Cap preview size for the chat response payload.
        state["table_preview"] = rows[:_MAX_TABLE_ROWS]
        title = result.get("title") or _TABLE_TITLES[tool_name]
        state["table_title"] = (
            f"{title} · first {_MAX_TABLE_ROWS} of {len(rows)} rows"
            if len(rows) > _MAX_TABLE_ROWS else title
        )
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
            "split_strategy": result.get("split_strategy"),
            "test_size": result.get("test_size"),
            "hyperparameters": result.get("hyperparameters"),
            "cv": result.get("cv"),
            "train": result.get("train"),
            "validation": result.get("validation"),
            "n_validation": result.get("n_validation"),
            "split": result.get("split"),
            "tested": result.get("tested"),
            "test": result.get("test"),
        }
        if "new_version_id" in result:
            state["active_dataset_version_id"] = result["new_version_id"]
        state["ui_update_required"] = True


def _new_ui_state(dataset_version_id: str, ui: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    ui = ui or {}
    return {
        "ui_update_required": False,
        "active_dataset_version_id": dataset_version_id,
        "plot_data": None,
        "table_preview": None,
        "table_title": None,
        "model_metrics": None,
        "selected_features": ui.get("selectedFeatures") or [],
        "target_variable": ui.get("targetVariable"),
    }


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


async def run_tool_manually(
    tool_name: str,
    arguments: dict[str, Any],
    dataset_version_id: str,
) -> dict[str, Any]:
    """Run one tool the user configured by hand.

    Goes through the same dispatch and UI-state folding as the chat agent, so
    a manual run updates the workbench exactly like the assistant calling it.
    The caller must already have checked that the user owns the version.
    """
    result = await _dispatch_tool(tool_name, arguments, dataset_version_id)
    state = _new_ui_state(dataset_version_id)
    _update_ui_state(state, tool_name=tool_name, result=result)
    message = result[1] if isinstance(result, tuple) and len(result) == 2 else _summarize_tool_result(tool_name, result)
    return {
        "tool": tool_name,
        "message": message,
        "summary": _summarize_tool_result(tool_name, result),
        "ui_update_required": bool(state["ui_update_required"]),
        "active_dataset_version_id": state["active_dataset_version_id"],
        "plot_data": state["plot_data"],
        "table_preview": state["table_preview"],
        "table_title": state["table_title"],
        "model_metrics": state["model_metrics"],
        "tool_calls_made": [tool_name],
    }


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


def _history_tools(history: list[Any]) -> list[str]:
    out: list[str] = []
    for turn in history:
        tools = turn.get("tools") if isinstance(turn, dict) else getattr(turn, "tools", None)
        out.extend(t for t in tools or [] if isinstance(t, str))
    return out


def _recent_user_texts(history: list[Any], n: int = 2) -> list[str]:
    texts = []
    for turn in history:
        role = turn.get("role") if isinstance(turn, dict) else getattr(turn, "role", None)
        content = turn.get("content") if isinstance(turn, dict) else getattr(turn, "content", None)
        if role == "user" and isinstance(content, str):
            texts.append(content)
    return texts[-n:]


def _reasoning_delta(delta: Any) -> Optional[str]:
    # vLLM >= 0.13 names the field `reasoning`; older servers use `reasoning_content`.
    return getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)


async def _stream_completion(
    client: Any,
    messages: list[dict[str, Any]],
    max_tokens: int,
    sink: dict[str, Any],
    think: bool = False,
    tools: Optional[list[dict[str, Any]]] = None,
) -> AsyncGenerator[tuple[str, str], None]:
    """Stream one model turn.

    Yields ``("reasoning", text)`` and ``("token", text)`` deltas and collects
    content, reasoning and tool calls in ``sink``.
    """
    stream = await client.chat.completions.create(
        temperature=0.1,
        max_tokens=max_tokens,
        tools=tools if tools is not None else _ALL_TOOL_SCHEMAS,
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
        "context_length", "context length", "maximum context", "reduce the length",
        "too large", "per-request limit", "tokens per minute", "rate limit",
        "internal server error", "upstream error", "413", "429", "500", "502", "503", "504", "529",
    ))


def _api_error_message(err: Exception) -> str:
    failures = getattr(err, "failures", None)
    if not failures:
        return f"I encountered an API error: {str(err)[:300]}. Please try rephrasing your request."
    lines = "\n".join(f"- {f}" for f in failures)
    return (
        f"No model could take this request:\n{lines}\n\n"
        "Long multi-step runs send a lot of context each round. They work best on the local model, "
        "which has no per-minute limit; on the free cloud tiers, try again in a minute or ask for "
        "one step at a time."
    )


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
    ui_state = _new_ui_state(dataset_version_id, ui)
    features_changed = False
    tool_calls_made: list[str] = []

    yield {"event": "status", "data": "Reading dataset schema and trained models"}
    ctx = await load_dataset_context(dataset_version_id)

    history = list(history or [])
    tool_history = render_tool_history(history)
    earlier_tools = _history_tools(history)
    request_text = " ".join([*_recent_user_texts(history), user_input])
    user_msg = {"role": "user", "content": user_input}
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": ""},
        *history_messages(history),
        user_msg,
    ]

    final_message = ""
    previous_tool_calls_str = None
    previous_round_failed = False
    repeat_nudged = False
    claim_checked = False

    for round_no in range(MAX_TOOL_ROUNDS):
        _shrink_seen_results(messages, next(i for i, m in enumerate(messages) if m is user_msg) + 1)
        if ctx is None or ctx.version_id != ui_state["active_dataset_version_id"]:
            ctx = await load_dataset_context(ui_state["active_dataset_version_id"])
        tools, omitted = select_schemas(
            _ALL_TOOL_SCHEMAS, user_input=request_text, step_hint=step_hint,
            used=[*earlier_tools, *tool_calls_made],
        )
        messages[0] = {
            "role": "system",
            "content": _execution_system_prompt(
                ctx, {**ui, "selectedFeatures": ui_state["selected_features"],
                      "targetVariable": ui_state["target_variable"]}, step_hint, tool_history, omitted,
            ),
        }
        yield {"event": "status", "data": "Reviewing tool results" if tool_calls_made else "Planning"}

        sink: dict[str, Any] = {"content": "", "reasoning": "", "tool_calls": {}}
        try:
            async for kind, text in _stream_completion(client, messages, 4096, sink, think, tools):
                yield {"event": kind, "data": text}
        except Exception as api_err:  # noqa: BLE001
            if not _is_recoverable(api_err) or sink["content"]:
                final_message = _api_error_message(api_err)
                yield {"event": "token", "data": final_message}
                break
            # Too large, rate-limited or an upstream crash: drop earlier chat turns and
            # shorten this turn's tool results, keeping its progress, and retry once.
            yield {"event": "status", "data": "Model call failed — retrying with a shorter context"}
            messages = _tight_messages(messages, user_msg)
            sink = {"content": "", "reasoning": "", "tool_calls": {}}
            try:
                async for kind, text in _stream_completion(client, messages, 2048, sink, think, tools):
                    yield {"event": kind, "data": text}
            except Exception as retry_err:  # noqa: BLE001
                final_message = _api_error_message(retry_err)
                yield {"event": "token", "data": final_message}
                break

        current_content = sink["content"]
        tool_calls = list(sink["tool_calls"].values())

        # Break out of infinite tool loops if the exact same tools and args are called sequentially
        current_tool_calls_str = json.dumps([
            {"name": tc["function"]["name"], "args": tc["function"]["arguments"]}
            for tc in tool_calls
        ], sort_keys=True)

        repeated = bool(tool_calls) and current_tool_calls_str == previous_tool_calls_str
        if repeated and (previous_round_failed or repeat_nudged):
            yield {"event": "thought", "data": current_content.strip()}
            final_message = (
                "*I stopped because I was repeating the same failing action. The tool errors are in the activity log above.*"
                if previous_round_failed
                else "*I stopped because I kept repeating the same step instead of moving on. The results so far are in the activity log above.*"
            )
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

        if repeated:
            # The same successful call again: answer it from the earlier result, once.
            repeat_nudged = True
            for i, tc in enumerate(tool_calls):
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"] or f"call_{round_no}_{i}",
                    "content": json.dumps({"note": _REPEAT_NOTE}),
                })
            continue

        if not tool_calls:
            if (
                not tool_calls_made
                and not claim_checked
                and _ACTION_REQUEST.search(user_input)
                and _ACTION_CLAIM.search(current_content)
            ):
                claim_checked = True
                yield {"event": "retract", "data": ""}
                messages.append({"role": "user", "content": _NO_TOOL_NUDGE})
                continue
            final_message = current_content.strip()
            break

        yield {"event": "thought", "data": current_content.strip()}

        previous_round_failed = False
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
                end = {
                    "ok": True,
                    "summary": _summarize_tool_result(name, result),
                    # Full arguments, so the UI can show the call in the tool's form.
                    "arguments": {k: v for k, v in (arguments or {}).items() if k != "dataset_version_id"},
                }
            except Exception as exc:  # noqa: BLE001 — surface tool errors to the LLM
                payload = _clip_result(json.dumps({"error": str(exc)}), _MAX_RESULT_CHARS)
                end = {"ok": False, "summary": _truncate(str(exc), _MAX_SUMMARY_CHARS)}
                previous_round_failed = True
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
            "table_title": ui_state["table_title"],
            "model_metrics": ui_state["model_metrics"],
            "selected_features": ui_state["selected_features"] if features_changed else None,
            "target_variable": ui_state["target_variable"] if features_changed else None,
            "tool_calls_made": tool_calls_made,
        }
    }
