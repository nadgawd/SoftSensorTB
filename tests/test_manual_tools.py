"""Manual mode: every tool and parameter the assistant has is a form control, and runs the same way."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from backend.api.tools import MANUAL_TOOLS, STEPS, manual_tool_specs
from backend.limits import manual_limiter
from backend.mcp_server import LLM_TOOL_SCHEMAS, TOOL_FUNCTIONS
from backend.mcp_server.eda_tools import (
    FeatureEngineerArgs,
    NormalizeDataArgs,
    RemoveOutliersArgs,
    _binary_encode,
    _ordinal_codes,
    _smote,
    _target_encode,
)
from backend.mcp_server.modeling_tools import TrainSoftSensorArgs, _split
from test_online_api import as_user, client, new_user, storage, upload  # noqa: F401 — fixtures

_VID = "00000000-0000-4000-8000-000000000000"
_SCHEMAS = {s["function"]["name"]: s["function"]["parameters"] for s in LLM_TOOL_SCHEMAS}
_SPECS = {s["name"]: s for s in manual_tool_specs()}


def _csv() -> bytes:
    rng = np.random.default_rng(0)
    n = 40
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": 2 * x1 - x2 + rng.normal(scale=0.1, size=n)})
    df.loc[[3, 7], "x2"] = np.nan
    return df.to_csv(index=False).encode()


def _run(client, user, version, tool, **arguments):  # noqa: F811
    return client.post(f"/api/tools/{tool}/run", headers=as_user(user),
                       json={"dataset_version_id": version, "arguments": arguments})


# ── The form spec covers everything the assistant can do ─────────────────


def test_every_tool_has_a_manual_form():
    assert set(MANUAL_TOOLS) == set(TOOL_FUNCTIONS)
    for name, ui in MANUAL_TOOLS.items():
        assert ui["steps"] and set(ui["steps"]) <= set(STEPS), name


@pytest.mark.parametrize("name", sorted(TOOL_FUNCTIONS))
def test_schema_and_function_take_the_same_parameters(name):
    """A parameter added to only one side would be unreachable from the form or the assistant."""
    schema_params = set(_SCHEMAS[name].get("properties", {}))
    fn_params = {
        p for p in inspect.signature(TOOL_FUNCTIONS[name]).parameters
        if p not in {"session", "dataset_version_id"}
    }
    assert schema_params == fn_params


@pytest.mark.parametrize("name", sorted(MANUAL_TOOLS))
def test_form_overrides_name_real_fields_and_options(name):
    ui = MANUAL_TOOLS[name]
    fields = {f["name"]: f for f in _SPECS[name]["fields"]}
    assert list(fields) == list(_SCHEMAS[name].get("properties", {}))
    for key in ("numeric", "advanced"):
        assert set(ui.get(key, ())) <= set(fields), (name, key)
    for key in ("defaults", "app_defaults", "show_when"):
        assert set(ui.get(key, {})) <= set(fields), (name, key)
    for field, conditions in ui.get("show_when", {}).items():
        for controller, values in conditions.items():
            assert set(values) <= set(fields[controller]["options"]), (name, field, controller)
    for field, default in ui.get("defaults", {}).items():
        if fields[field]["widget"] == "select":
            assert default in fields[field]["options"], (name, field)


def test_every_required_field_has_a_default_or_is_a_user_choice():
    choices = {"column", "columns", "mapping", "text", "code"}
    for spec in _SPECS.values():
        for f in spec["fields"]:
            if f["required"] and "default" not in f and "app_default" not in f:
                assert f["widget"] in choices, (spec["name"], f["name"])


def test_train_test_split_ratio_is_a_parameter():
    fields = {f["name"]: f for f in _SPECS["train_soft_sensor"]["fields"]}
    assert fields["test_size"]["default"] == 0.2
    assert (fields["test_size"]["min"], fields["test_size"]["max"]) == (0.05, 0.5)
    assert "test_size" in _SCHEMAS["train_soft_sensor"]["properties"]


# ── Parameters behave as documented ──────────────────────────────────────


@pytest.mark.parametrize("strategy", ["random", "sequential"])
@pytest.mark.parametrize("test_size", [0.1, 0.25, 0.5])
def test_split_honours_test_size(strategy, test_size):
    X = pd.DataFrame({"a": range(100)})
    y = pd.Series(range(100))
    args = TrainSoftSensorArgs(dataset_version_id=_VID, algorithm="OLS", target_column="y",
                               feature_columns=["a"], split_strategy=strategy, test_size=test_size)
    X_train, X_test, _, _ = _split(X, y, args)
    assert len(X_test) == int(100 * test_size)
    assert len(X_train) + len(X_test) == 100
    if strategy == "sequential":
        assert X_test.index.min() > X_train.index.max()


def test_split_ratio_outside_limits_is_rejected():
    for bad in (0.0, 0.9):
        with pytest.raises(ValidationError):
            TrainSoftSensorArgs(dataset_version_id=_VID, algorithm="OLS", target_column="y",
                                feature_columns=["a"], test_size=bad)


def test_encoders():
    df = pd.DataFrame({"c": ["b", "a", "c", None, "a"], "y": [10.0, 0.0, 6.0, 1.0, 2.0]})
    assert _ordinal_codes(df["c"]).tolist()[:3] == [2.0, 1.0, 3.0]
    assert np.isnan(_ordinal_codes(df["c"])[3])

    binary = _binary_encode(df, "c")
    assert list(binary.columns) == ["c_0", "c_1", "y"]
    assert binary[["c_0", "c_1"]].values.tolist() == [[1, 0], [0, 1], [1, 1], [0, 0], [0, 1]]

    plain = _target_encode(df["c"], df["y"], smoothing=0)
    assert plain.tolist() == [10.0, 1.0, 6.0, df["y"].mean(), 1.0]
    smoothed = _target_encode(df["c"], df["y"], smoothing=2)
    overall = df["y"].mean()
    assert smoothed[0] == pytest.approx((10 + 2 * overall) / 3)


def test_smote_balances_with_points_between_neighbours():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "a": np.r_[rng.normal(0, 1, 40), rng.normal(10, 1, 8)],
        "tag": ["x"] * 48,
        "label": [0] * 40 + [1] * 8,
    })
    out = _smote(df, "label", df["label"].value_counts(), 3, np.random.default_rng(1))
    assert out["label"].value_counts().to_dict() == {0: 40, 1: 40}
    minority = out[out["label"] == 1]["a"]
    original = df[df["label"] == 1]["a"]
    assert minority.between(original.min(), original.max()).all()
    assert (out["tag"] == "x").all()
    with pytest.raises(ValueError, match="k_neighbors"):
        _smote(df, "label", df["label"].value_counts(), 8, np.random.default_rng(1))


def test_cross_field_validation():
    with pytest.raises(ValidationError, match="below 0.5"):
        RemoveOutliersArgs(dataset_version_id=_VID, columns=["a"], method="winsorize", threshold=3)
    with pytest.raises(ValidationError, match="feature_min"):
        NormalizeDataArgs(dataset_version_id=_VID, columns=["a"], strategy="min_max", feature_min=1, feature_max=0)
    with pytest.raises(ValidationError, match="exactly 2"):
        FeatureEngineerArgs(dataset_version_id=_VID, columns=["a"], strategy="ratio")


# ── The run endpoint ─────────────────────────────────────────────────────


def test_manual_spec_is_served(client):  # noqa: F811
    body = client.get("/api/tools/manual").json()
    assert body["steps"] == list(STEPS)
    assert {t["name"] for t in body["tools"]} == set(TOOL_FUNCTIONS)


def test_schema_endpoint_lists_columns(client):  # noqa: F811
    alice = new_user()
    version = upload(client, alice, body=_csv())["dataset_version_id"]
    body = client.get(f"/datasets/versions/{version}/schema", headers=as_user(alice)).json()
    assert [(c["name"], c["numeric"]) for c in body["columns"]] == [("x1", True), ("x2", True), ("y", True)]
    assert body["row_count"] == 40 and body["models"] == []
    assert client.get(f"/datasets/versions/{version}/schema", headers=as_user(new_user())).status_code == 404


def test_manual_pipeline_end_to_end(client):  # noqa: F811
    alice = new_user()
    version = upload(client, alice, body=_csv())["dataset_version_id"]

    stats = _run(client, alice, version, "get_column_statistics")
    assert stats.status_code == 200, stats.text
    rows = {r["column"]: r for r in stats.json()["table_preview"]}
    assert rows["x2"]["null_count"] == 2
    assert stats.json()["table_title"] == "Column statistics"

    imputed = _run(client, alice, version, "remove_missing_data", strategy="knn", n_neighbors=3, columns=["x2"])
    assert imputed.status_code == 200, imputed.text
    new_version = imputed.json()["active_dataset_version_id"]
    assert new_version != version
    assert "null cells 2 → 0" in imputed.json()["message"]
    history = client.get(f"/datasets/{upload_dataset_id(client, alice, new_version)}/history", headers=as_user(alice)).json()
    params = next(v["parameters_used"] for v in history if v["id"] == new_version)
    assert params["n_neighbors"] == 3 and params["columns"] == ["x2"]

    trained = _run(client, alice, new_version, "train_soft_sensor", algorithm="RIDGE", target_column="y",
                   feature_columns=["x1", "x2"], alpha=0.5, test_size=0.25, random_state=7)
    assert trained.status_code == 200, trained.text
    metrics = trained.json()["model_metrics"]
    assert (metrics["n_train"], metrics["n_test"]) == (30, 10)
    assert metrics["hyperparameters"] == {"split_strategy": "random", "test_size": 0.25, "random_state": 7,
                                          "alpha": 0.5, "fit_intercept": True}
    assert metrics["r2_score"] > 0.9

    scored = trained.json()["active_dataset_version_id"]
    parity = _run(client, alice, scored, "generate_parity_plot", model_id=metrics["model_id"])
    assert parity.status_code == 200, parity.text
    assert parity.json()["plot_data"]["data"]

    models = _run(client, alice, scored, "list_trained_models")
    assert [r["model_id"] for r in models.json()["table_preview"]] == [metrics["model_id"]]


def _rich_csv() -> bytes:
    rng = np.random.default_rng(1)
    n = 60
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    df = pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "y": 2 * x1 - x2 + rng.normal(scale=0.1, size=n),
        "cat": [["a", "b", "c"][i % 3] for i in range(n)],
        "label": [1 if i % 6 == 0 else 0 for i in range(n)],
        "hour": [i % 24 for i in range(n)],
        "ts": pd.date_range("2026-01-01", periods=n, freq="h").astype(str),
    })
    df.loc[[3, 7], "x2"] = np.nan
    return df.to_csv(index=False).encode()


# Columns a user would pick for each tool (and per option where it matters).
_PICKS = {
    "get_column_statistics": {},
    "run_duckdb_query": {},
    "remove_missing_data": {},
    "remove_outliers": {"columns": ["x1"]},
    "detect_anomalies": {},
    "drop_columns": {"columns": ["cat"]},
    "rename_columns": {"column_mapping": {"x1": "x1_renamed"}},
    "normalize_data": {"columns": ["x1", "x2"]},
    "transform_features": {"columns": ["x1"]},
    "encode_categorical": {"columns": ["cat"], "target_column": "y"},
    "rolling_aggregate": {"columns": ["x1"]},
    "create_lag_features": {"columns": ["x1"]},
    "feature_engineer": {"columns": ["x1", "x2"]},
    "execute_formula": {"new_column_name": "z", "formula": "x1 + y"},
    "balance_data": {"target_column": "label"},
    "generate_custom_plot": {"x_column": "x1", "y_columns": ["y"], "z_column": "x2"},
    "select_features_rfe": {"target_column": "y", "n_features_to_select": 2},
    "drop_collinear_features": {"target_column": "y"},
    "reduce_dimensions": {"columns": ["x1", "x2"]},
    "train_soft_sensor": {"target_column": "y", "feature_columns": ["x1", "x2"]},
    "list_trained_models": {},
}
_PER_OPTION = {
    ("feature_engineer", "datetime"): {"columns": ["ts"]},
    ("feature_engineer", "cyclical"): {"columns": ["hour"]},
}


def _main_choice(spec):
    return next((f for f in spec["fields"] if f["widget"] == "select" and f["required"]), None)


def _visible(field, values):
    return all(values.get(k) in v for k, v in (field.get("show_when") or {}).items())


def _form_arguments(spec, option=None):
    """What the form sends untouched: its defaults, for the fields shown, plus the user's picks."""
    main = _main_choice(spec)
    values = {f["name"]: f["default"] for f in spec["fields"] if "default" in f}
    if main and option is not None:
        values[main["name"]] = option
    values.update(_PICKS[spec["name"]])
    values.update(_PER_OPTION.get((spec["name"], option), {}))
    fields = {f["name"]: f for f in spec["fields"]}
    return {k: v for k, v in values.items() if k in fields and v is not None and _visible(fields[k], values)}


_CASES = [
    (name, option)
    for name, spec in _SPECS.items()
    if name in _PICKS
    for option in ((_main_choice(spec) or {}).get("options") or [None])
]


@pytest.mark.parametrize(("name", "option"), _CASES, ids=[f"{n}-{o}" for n, o in _CASES])
def test_every_tool_and_option_runs_from_its_form_defaults(client, name, option):  # noqa: F811
    alice = new_user()
    version = upload(client, alice, body=_rich_csv(), name="rich.csv")["dataset_version_id"]
    resp = _run(client, alice, version, name, **_form_arguments(_SPECS[name], option))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ui_update_required"]
    kind = _SPECS[name]["kind"]
    if kind == "plot":
        assert body["plot_data"]["data"]
    elif kind == "train":
        assert body["model_metrics"]["n_test"] > 0
    elif kind == "table":
        assert isinstance(body["table_preview"], list)


@pytest.mark.parametrize("name", [n for n, ui in MANUAL_TOOLS.items() if ui["steps"] == ("validation",)])
def test_diagnostic_plots_run_from_their_form(client, name):  # noqa: F811
    alice = new_user()
    version = upload(client, alice, body=_rich_csv(), name="rich.csv")["dataset_version_id"]
    trained = _run(client, alice, version, "train_soft_sensor", **_form_arguments(_SPECS["train_soft_sensor"], "OLS"))
    assert trained.status_code == 200, trained.text
    # The model picker defaults to "latest model", which sends no model_id.
    resp = _run(client, alice, trained.json()["active_dataset_version_id"], name)
    assert resp.status_code == 200, resp.text
    assert resp.json()["plot_data"]["data"]


def upload_dataset_id(client, user, version):  # noqa: F811
    return client.get(f"/datasets/versions/{version}/meta", headers=as_user(user)).json()["dataset_id"]


def test_bad_arguments_come_back_as_readable_errors(client):  # noqa: F811
    alice = new_user()
    version = upload(client, alice, body=_csv())["dataset_version_id"]

    invalid = _run(client, alice, version, "train_soft_sensor", algorithm="OLS", target_column="y",
                   feature_columns=["x1"], test_size=0.9)
    assert invalid.status_code == 422
    assert "test_size" in invalid.json()["detail"]

    missing = _run(client, alice, version, "normalize_data", columns=["nope"], strategy="standard")
    assert missing.status_code == 400
    assert "nope" in missing.json()["detail"]

    unknown_arg = _run(client, alice, version, "drop_columns", columns=["x1"], session="x")
    assert unknown_arg.status_code == 200  # `session` is never forwarded


def test_manual_runs_are_owner_only(client):  # noqa: F811
    alice, bob = new_user(), new_user()
    version = upload(client, alice, body=_csv())["dataset_version_id"]
    assert _run(client, bob, version, "drop_columns", columns=["x1"]).status_code == 404
    assert _run(client, alice, version, "not_a_tool").status_code == 404
    assert _run(client, alice, version, "set_model_features").status_code == 404


def test_manual_runs_are_rate_limited(client, monkeypatch):  # noqa: F811
    monkeypatch.setenv("MANUAL_RATE_LIMIT_PER_HOUR", "1")
    manual_limiter.reset()
    alice = new_user()
    version = upload(client, alice, body=_csv())["dataset_version_id"]
    assert _run(client, alice, version, "get_column_statistics").status_code == 200
    assert _run(client, alice, version, "get_column_statistics").status_code == 429
    manual_limiter.reset()
