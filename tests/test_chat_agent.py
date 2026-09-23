"""Chat agent grounding: schema/model context, history, feature selection, event stream."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from backend.agents import execution_agent
from backend.agents.context import (
    DatasetContext,
    history_messages,
    render_dataset_context,
    render_tool_history,
)
from backend.agents.router import classify_intent
from backend.agents.ui_tools import apply_feature_selection
from backend.mcp_server.model_registry import models_for_lineage

COLUMNS = ["DT", "TI1", "T1", "WF", "T2", "S", "TI1_lag1", "T2_lag1", "TI1_rolling5_mean"]
NUMERIC = [c for c in COLUMNS if c != "DT"]


def select(**kwargs):
    base = dict(columns=COLUMNS, numeric_columns=NUMERIC, current_features=["TI1", "T1", "WF", "T2"], current_target="S")
    return apply_feature_selection(**{**base, **kwargs})


class TestFeatureSelection:
    def test_select_replaces_with_wildcards_in_column_order(self):
        r = select(select=["*rolling*", "*_lag*"])
        assert r["selected_features"] == ["TI1_lag1", "T2_lag1", "TI1_rolling5_mean"]
        assert r["now_unselected"] == ["TI1", "T1", "WF", "T2"]
        assert r["now_selected"] == ["TI1_lag1", "T2_lag1", "TI1_rolling5_mean"]
        assert "warning" not in r

    def test_add_and_remove_edit_current_selection(self):
        r = select(add=["ti1_lag1"], remove=["T1"])
        assert r["selected_features"] == ["TI1", "WF", "T2", "TI1_lag1"]

    def test_target_and_non_numeric_are_never_features(self):
        r = select(select=["S", "DT", "TI1"])
        assert r["selected_features"] == ["TI1"]
        assert r["ignored_non_numeric"] == ["DT"]

    def test_changing_target_drops_it_from_features(self):
        r = select(target_column="T2")
        assert r["target_column"] == "T2"
        assert "T2" not in r["selected_features"]

    def test_unknown_names_reported_or_rejected(self):
        assert select(add=["TI1_lag1", "sensor1_value"])["ignored_unknown"] == ["sensor1_value"]
        with pytest.raises(ValueError, match="No such columns"):
            select(add=["sensor1_value"])

    def test_empty_selection_warns_so_the_model_retries(self):
        r = select(remove=["TI1", "T1", "WF", "T2"])
        assert r["selected_features"] == []
        assert "select" in r["warning"]


def test_models_for_lineage_matches_ancestors_newest_first():
    metas = [
        {"model_id": "a", "dataset_version_id": "v1", "trained_at": "2026-01-01"},
        {"model_id": "b", "dataset_version_id": "v2", "trained_at": "2026-03-01"},
        {"model_id": "c", "dataset_version_id": "other", "new_version_id": "v3", "trained_at": "2026-02-01"},
        {"model_id": "d", "dataset_version_id": "unrelated", "trained_at": "2026-04-01"},
    ]
    assert [m["model_id"] for m in models_for_lineage(metas, ["v3", "v2", "v1"])] == ["b", "c", "a"]


def test_dataset_context_lists_real_columns_and_models():
    ctx = DatasetContext(
        version_id="v1", action="normalize_data", n_rows=300,
        columns=[("DT", "string", False), ("TI1", "double", True), ("S", "double", True)],
        models=[{"model_id": "m1", "algorithm": "PLS", "target_column": "S", "feature_columns": ["TI1"], "r2_score": 0.75, "rmse": 0.42}],
    )
    text = render_dataset_context(ctx)
    assert "  - TI1 (double)" in text
    assert "NUMERIC COLUMNS: TI1, S" in text
    assert "m1: PLS predicting S" in text
    assert "schema unavailable" in render_dataset_context(None)


def test_history_messages_keeps_recent_turns_and_tool_names_out_of_band():
    turns = [{"role": "user", "content": f"q{i}"} for i in range(20)]
    turns.append({"role": "assistant", "content": "made lags", "tools": ["create_lag_features"]})
    turns.append({"role": "system", "content": "ignored"})
    turns.append(SimpleNamespace(role="user", content="now rolling mean", tools=[]))
    out = history_messages(turns)
    assert len(out) <= 12
    assert out[-1] == {"role": "user", "content": "now rolling mean"}
    # Tool names go to the system prompt: the model copies annotations it sees in its own turns.
    assert out[-2] == {"role": "assistant", "content": "made lags"}
    assert all(m["role"] in ("user", "assistant") for m in out)
    assert "reply 1: create_lag_features" in render_tool_history(turns)
    assert render_tool_history([{"role": "user", "content": "hi"}]) == ""


@pytest.mark.parametrize(
    "text,previous,expected",
    [
        ("unselect the old features", None, "EXECUTE"),
        ("go ahead", "EXECUTE", "EXECUTE"),
        ("what is PLS regression?", "EXECUTE", "RAG"),
    ],
)
def test_router_follow_ups(text, previous, expected, monkeypatch):
    async def no_llm(*_a, **_k):
        raise AssertionError("router LLM must not be called")
    monkeypatch.setattr("backend.agents.router.get_llm_client", lambda *a, **k: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=no_llm))))
    assert asyncio.run(classify_intent(text, has_dataset=True, previous_intent=previous)) == expected


def test_training_on_lagged_data_keeps_leading_rows_as_nan(monkeypatch, tmp_path):
    import numpy as np
    import pandas as pd

    from backend.mcp_server import modeling_tools as mt

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"TI1": rng.normal(size=50), "S": rng.normal(size=50)})
    df["TI1_lag1"] = df["TI1"].shift(1)
    df["TI1_lag2"] = df["TI1"].shift(2)
    persisted = {}

    async def fake_get_version(_db, _vid):
        return SimpleNamespace(id="v1", dataset_id="d1")

    async def fake_persist(_db, *, df, **_kw):
        persisted["df"] = df.copy()
        return "v2", "ok"

    async def fake_register(_db, **kw):
        persisted["key"] = kw["artifact_key"]

    async def fake_get(_model, _id):
        return SimpleNamespace(id="d1", owner_id="u1")

    monkeypatch.setattr(mt, "_get_version", fake_get_version)
    monkeypatch.setattr(mt, "_load_dataframe", lambda _v: df.copy())
    monkeypatch.setattr(mt, "_persist_new_version", fake_persist)
    monkeypatch.setattr(mt, "_save_model_artifact", lambda key, _a: key)
    monkeypatch.setattr(mt, "register_model", fake_register)

    result = asyncio.run(mt.train_soft_sensor(
        dataset_version_id="00000000-0000-0000-0000-000000000001",
        target_column="S", feature_columns=["TI1_lag1", "TI1_lag2"], algorithm="RIDGE",
        session=SimpleNamespace(get=fake_get),
    ))
    assert persisted["key"] == f"u1/d1/models/{result['model_id']}.pkl"
    pred = persisted["df"]["Predicted_S"]
    assert len(pred) == 50
    assert pred.iloc[:2].isna().all() and pred.iloc[2:].notna().all()
    assert result["n_train"] + result["n_test"] == 48


# ── Execution agent event stream ───────────────────────────────────────────


def _chunk(content=None, tool=None, reasoning=None):
    tool_calls = None
    if tool:
        idx, call_id, name, args = tool
        tool_calls = [SimpleNamespace(index=idx, id=call_id, function=SimpleNamespace(name=name, arguments=args))]
    delta = SimpleNamespace(content=content, tool_calls=tool_calls, reasoning=reasoning)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


class ScriptedClient:
    """Replays one list of chunks per model round and records the prompts."""

    def __init__(self, rounds):
        self.rounds = list(rounds)
        self.prompts = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.prompts.append([dict(m) for m in kwargs["messages"]])
        self.think = kwargs.get("think")
        chunks = self.rounds.pop(0)

        async def gen():
            for c in chunks:
                yield c
        return gen()


def test_execution_agent_streams_steps_and_applies_feature_selection(monkeypatch):
    client = ScriptedClient([
        [_chunk("Checking the lag columns."), _chunk(tool=(0, "c1", "run_duckdb_query", '{"query": "SELECT 1"}'))],
        [_chunk(tool=(0, "c2", "set_model_features", '{"select": ["*_lag*", "*rolling*"]}'))],
        [_chunk("Selected 3 features "), _chunk("for S.")],
    ])
    ctx = DatasetContext("v1", "rolling_aggregate", 300, [(c, "double", c in NUMERIC) for c in COLUMNS])
    dispatched = []

    async def fake_dispatch(name, args, version_id):
        dispatched.append((name, args, version_id))
        return {"rows": [{"x": 1}]}

    async def fake_ctx(_vid):
        return ctx

    monkeypatch.setattr(execution_agent, "get_llm_client", lambda **_k: client)
    monkeypatch.setattr(execution_agent, "load_dataset_context", fake_ctx)
    monkeypatch.setattr(execution_agent, "_dispatch_tool", fake_dispatch)

    async def collect():
        return [e async for e in execution_agent.run_execution_agent(
            "select only the new features",
            "v1",
            ui_context=json.dumps({"targetVariable": "S", "selectedFeatures": ["TI1", "T1"]}),
            history=[{"role": "user", "content": "make lag features"},
                     {"role": "assistant", "content": "Done", "tools": ["create_lag_features"]}],
        )]

    events = asyncio.run(collect())
    kinds = [e["event"] for e in events]

    assert kinds.count("tool_start") == 2 and kinds.count("tool_end") == 2
    assert {"event": "thought", "data": "Checking the lag columns."} in events
    assert dispatched == [("run_duckdb_query", {"query": "SELECT 1"}, "v1")]

    ends = [e["data"] for e in events if e["event"] == "tool_end"]
    assert all(end["ok"] for end in ends)
    assert "3 features" in ends[1]["summary"]

    final = events[-1]
    assert final["event"] == "final_state"
    assert final["data"]["selected_features"] == ["TI1_lag1", "T2_lag1", "TI1_rolling5_mean"]
    assert final["data"]["target_variable"] == "S"
    assert final["data"]["tool_calls_made"] == ["run_duckdb_query", "set_model_features"]

    first_prompt = client.prompts[0]
    assert "TI1_rolling5_mean (double)" in first_prompt[0]["content"]
    assert "selected features: ['TI1', 'T1']" in first_prompt[0]["content"]
    assert [m["role"] for m in first_prompt[1:]] == ["user", "assistant", "user"]
    assert "reply 1: create_lag_features" in first_prompt[0]["content"]
    # After the selection tool runs, the next round sees the updated UI state.
    assert "TI1_lag1" in client.prompts[2][0]["content"].split("UI STATE")[1]


def test_execution_agent_streams_reasoning_and_replays_it(monkeypatch):
    client = ScriptedClient([
        [_chunk(reasoning="The user wants lags. "), _chunk(reasoning="Query first."),
         _chunk(tool=(0, "c1", "run_duckdb_query", '{"query": "SELECT 1"}'))],
        [_chunk(reasoning="One row came back."), _chunk("There is 1 row.")],
    ])
    ctx = DatasetContext("v1", "raw", 10, [(c, "double", True) for c in ("T1", "S")])

    async def fake_dispatch(name, args, version_id):
        return {"rows": [{"x": 1}]}

    async def fake_ctx(_vid):
        return ctx

    monkeypatch.setattr(execution_agent, "get_llm_client", lambda **_k: client)
    monkeypatch.setattr(execution_agent, "load_dataset_context", fake_ctx)
    monkeypatch.setattr(execution_agent, "_dispatch_tool", fake_dispatch)

    async def collect():
        return [e async for e in execution_agent.run_execution_agent("how many rows?", "v1", think=True)]

    events = asyncio.run(collect())

    assert client.think is True
    reasoning = [e["data"] for e in events if e["event"] == "reasoning"]
    assert reasoning == ["The user wants lags. ", "Query first.", "One row came back."]
    assert "".join(e["data"] for e in events if e["event"] == "token") == "There is 1 row."
    # Reasoning is shown in the activity log, never as part of the answer.
    assert events[-1]["event"] == "final_state"
    assert "Query first" not in events[-1]["data"].get("summary", "")

    # The second round replays the first round's reasoning with its tool call.
    replayed = [m for m in client.prompts[1] if m["role"] == "assistant"]
    assert replayed[-1]["reasoning_content"] == "The user wants lags. Query first."
    assert replayed[-1]["tool_calls"][0]["function"]["name"] == "run_duckdb_query"


def _run_agent(monkeypatch, client, message, history=None):
    ctx = DatasetContext("v1", "raw", 10, [(c, "double", True) for c in ("x", "y")])
    dispatched = []

    async def fake_dispatch(name, args, version_id):
        dispatched.append(name)
        return {"data": [], "layout": {"title": {"text": "Feature importance — PLS"}}}

    async def fake_ctx(_vid):
        return ctx

    monkeypatch.setattr(execution_agent, "get_llm_client", lambda **_k: client)
    monkeypatch.setattr(execution_agent, "load_dataset_context", fake_ctx)
    monkeypatch.setattr(execution_agent, "_dispatch_tool", fake_dispatch)

    async def collect():
        return [e async for e in execution_agent.run_execution_agent(message, "v1", history=history)]

    return asyncio.run(collect()), dispatched


def test_claimed_plot_without_a_tool_call_is_retracted_and_redone(monkeypatch):
    client = ScriptedClient([
        [_chunk("**Feature Importance Plot Generated Successfully**\n\nThe plot is now displayed.")],
        [_chunk(tool=(0, "c1", "generate_importance_plot", "{}"))],
        [_chunk("Feature importance plot for PLS: x is the only input.")],
    ])
    history = [{"role": "user", "content": "Generate a parity plot"},
               {"role": "assistant", "content": "**Parity Plot Generated Successfully**",
                "tools": ["generate_parity_plot"]}]
    events, dispatched = _run_agent(monkeypatch, client, "Show the feature importance bar chart", history)
    kinds = [e["event"] for e in events]

    assert dispatched == ["generate_importance_plot"]
    assert kinds.index("retract") < kinds.index("tool_start")
    assert "[Automatic check]" in client.prompts[1][-1]["content"]
    final = events[-1]["data"]
    assert final["plot_data"] is not None and final["ui_update_required"] is True
    answer = "".join(e["data"] for e in events[kinds.index("tool_end"):] if e["event"] == "token")
    assert answer == "Feature importance plot for PLS: x is the only input."


def test_plain_answer_without_tools_is_not_second_guessed(monkeypatch):
    client = ScriptedClient([[_chunk("The target is y and the only selected feature is x.")]])
    events, dispatched = _run_agent(monkeypatch, client, "what is the target?")
    assert dispatched == []
    assert "retract" not in [e["event"] for e in events]
    assert len(client.prompts) == 1


def test_claim_check_runs_once_so_a_stubborn_model_cannot_loop(monkeypatch):
    claim = [_chunk("Plot generated and displayed.")]
    client = ScriptedClient([claim, claim])
    events, dispatched = _run_agent(monkeypatch, client, "plot it")
    assert [e["event"] for e in events].count("retract") == 1
    assert len(client.prompts) == 2
    assert events[-1]["event"] == "final_state"
