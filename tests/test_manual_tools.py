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
from backend.mcp_server.modeling_tools import TrainSoftSensorArgs, _cross_validate, _split, _strata
from test_online_api import as_user, client, new_user, storage, upload  # noqa: F401 — fixtures

_VID = "00000000-0000-4000-8000-000000000000"


def _alternatives(when):
    """``show_when`` is one dict of conditions, or a list of dicts of which any may hold."""
    return when if isinstance(when, list) else [when]
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
    for field, when in ui.get("show_when", {}).items():
        for conditions in _alternatives(when):
            for controller, values in conditions.items():
                assert set(values) <= set(fields[controller]["options"]), (name, field, controller)
    for field, default in ui.get("defaults", {}).items():
        if fields[field]["widget"] == "select":
            assert default in fields[field]["options"], (name, field)


def test_every_required_field_has_a_default_or_is_a_user_choice():
    choices = {"column", "columns", "mapping", "text", "code", "dataset", "value_mapping"}
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
        "country": [["IND", "USA", "FRA", "BRA"][i % 4] for i in range(n)],
        "lat": 19.0 + rng.normal(scale=0.5, size=n),
        "lon": 72.8 + rng.normal(scale=0.5, size=n),
        "note": [["pump running smoothly today", "valve leak detected near inlet",
                  "sensor drift observed after cleaning", "pump vibration high near valve"][i % 4] for i in range(n)],
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
    "inspect_data": {},
    "drop_rows": {"column": "x1", "operator": "greater_than", "value": "1.5"},
    "drop_duplicates": {},
    "convert_types": {"columns": ["hour"]},
    "clean_text": {"columns": ["cat"]},
    "select_k_best": {"target_column": "y", "k": 2},
    "aggregate_data": {"group_by": ["cat"]},
    "statistical_test": {"column": "x1"},
    "list_datasets": {},
    "sort_data": {"by": ["x1"]},
    "combine_datasets": {"other_dataset": "rich.csv", "on": ["hour"]},
    "reshape_data": {"id_columns": ["cat"], "value_columns": ["x1", "x2"]},
    "mask_values": {"columns": ["x1"], "value": "0"},
    "map_values": {"columns": ["cat"], "value_mapping": {"a": "alpha", "b": "beta"}},
    "apply_function": {"columns": ["x1"], "group_by": ["cat"]},
    "vectorize_text": {"column": "note"},
    "geo_features": {"lat_column": "lat", "lon_column": "lon", "ref_lat": 19.07, "ref_lon": 72.88},
    "select_features_model": {"target_column": "y", "feature_columns": ["x1", "x2", "hour"], "n_features": 2},
}
_CATEGORY_PLOTS = ("bar", "count", "swarm", "strip", "point", "lollipop", "pie", "donut", "waterfall")
_MANY_COLUMN_PLOTS = ("parallel_coordinates", "radviz", "andrews_curves", "dendrogram", "network", "chord", "silhouette")
_PER_OPTION = {
    **{("generate_custom_plot", t): {"x_column": "cat"} for t in _CATEGORY_PLOTS},
    **{("generate_custom_plot", t): {"y_columns": ["x1", "x2", "y"]} for t in _MANY_COLUMN_PLOTS},
    ("generate_custom_plot", "bubble"): {"value_column": "hour"},
    ("generate_custom_plot", "treemap"): {"path_columns": ["cat", "country"]},
    ("generate_custom_plot", "sunburst"): {"path_columns": ["cat", "country"], "value_column": "hour"},
    ("generate_custom_plot", "sankey"): {"path_columns": ["cat", "country"]},
    ("generate_custom_plot", "choropleth"): {"location_column": "country", "value_column": "y"},
    ("generate_custom_plot", "bubble_map"): {"lat_column": "lat", "lon_column": "lon", "value_column": "hour"},
    ("generate_custom_plot", "density_map"): {"lat_column": "lat", "lon_column": "lon"},
    ("clean_text", "contains"): {"find": "a"},
    ("clean_text", "extract"): {"find": "([a-c])"},
    ("drop_rows", "query"): {"query": "x1 > 0 and hour < 20"},
    ("encode_categorical", "woe"): {"target_column": "label"},
    ("reduce_dimensions", "truncated_svd"): {"columns": ["x1", "x2", "y"]},
    ("reduce_dimensions", "autoencoder"): {"columns": ["x1", "x2", "y"]},
    ("reduce_dimensions", "lda"): {"target_column": "cat"},
    ("statistical_test", "t_test_independent"): {"column": "y", "group_column": "label"},
    ("statistical_test", "t_test_paired"): {"column_b": "x2"},
    ("statistical_test", "anova"): {"column": "y", "group_column": "cat"},
    ("statistical_test", "chi_square"): {"column": "cat", "column_b": "country"},
    ("reshape_data", "pivot"): {"id_columns": ["hour"], "pivot_column": "cat", "value_columns": ["y"]},
    ("reshape_data", "unstack"): {"id_columns": ["ts"], "pivot_column": "cat", "value_columns": ["y"]},
    ("apply_function", "sqrt"): {"columns": ["hour"]},
    ("apply_function", "log"): {"columns": ["lat"]},
    ("apply_function", "log10"): {"columns": ["lat"]},
    ("apply_function", "reciprocal"): {"columns": ["lat"]},
    ("apply_function", "cumprod"): {"columns": ["lat"]},
    ("apply_function", "exp"): {"columns": ["x1"]},

    ("feature_engineer", "datetime"): {"columns": ["ts"]},
    ("feature_engineer", "cyclical"): {"columns": ["hour"]},
    ("remove_outliers", "clip"): {"upper_bound": 1.0},
    ("drop_rows", "row_range"): {"end_row": 5},
    ("convert_types", "string"): {"columns": ["cat"]},
    ("convert_types", "category"): {"columns": ["cat"]},
    ("convert_types", "boolean"): {"columns": ["label"]},
    ("convert_types", "datetime"): {"columns": ["ts"]},
    ("clean_text", "replace"): {"find": "a"},
}


def _main_choice(spec):
    return next((f for f in spec["fields"] if f["widget"] == "select" and f["required"]), None)


def _visible(field, values):
    when = field.get("show_when")
    if not when:
        return True
    return any(all(values.get(k) in v for k, v in c.items()) for c in _alternatives(when))


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


# ── EDA, data manipulation, feature selection and split options ──────────


def _preview(client, user, version):  # noqa: F811
    body = client.get(f"/datasets/versions/{version}/preview?limit=1000", headers=as_user(user)).json()
    return pd.DataFrame(body["table_preview"], columns=body["columns"])


def _messy_csv() -> bytes:
    df = pd.DataFrame({
        "x": [1.0, 2.0, 2.0, 50.0, 3.0, -40.0, 2.5, 2.0],
        "y": [2.0, 4.1, 4.1, 9.0, 6.2, 1.0, 5.0, 4.1],
        "name": ["  Alpha ", "beta", "beta", "GAMMA  ray", "NA", "delta", "Beta", "beta"],
        "flag": ["yes", "no", "no", "Y", "n", "true", "?", "no"],
        "when": ["2026-01-05", "2026-01-06", "2026-01-06", "05/01/2026", "2026-01-08", "bad", "2026-01-10", "2026-01-06"],
        "level": ["low", "high", "high", "medium", "low", "medium", "high", "high"],
    })
    return df.to_csv(index=False).encode()


def _ok(resp):
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_inspect_views_cover_pandas_eda(client):  # noqa: F811
    alice = new_user()
    version = upload(client, alice, body=_messy_csv(), name="messy.csv")["dataset_version_id"]

    def view(**kw):
        body = _ok(_run(client, alice, version, "inspect_data", **kw))
        assert not body["ui_update_required"] or body["table_preview"] is not None
        assert body["active_dataset_version_id"] == version
        return body["table_title"], body["table_preview"]

    title, rows = view(view="head", n_rows=3)
    assert title.startswith("head") and [r["row"] for r in rows] == [0, 1, 2] and rows[0]["x"] == 1.0
    _, rows = view(view="tail", n_rows=2)
    assert [r["row"] for r in rows] == [6, 7]
    _, rows = view(view="shape")
    assert rows == [{"rows": 8, "columns": 6}]
    _, rows = view(view="info")
    info = {r["column"]: r for r in rows}
    assert info["name"]["missing"] == 1 and info["flag"]["missing"] == 1 and info["x"]["non_null"] == 8
    _, rows = view(view="missing")
    assert rows[0]["missing"] == 1 and {r["column"] for r in rows} == {"x", "y", "name", "flag", "when", "level"}
    _, rows = view(view="value_counts", columns=["level"], top_n=2)
    assert [(r["value"], r["count"]) for r in rows] == [("high", 4), ("low", 2)]
    assert rows[0]["distinct_values"] == 3
    _, rows = view(view="describe")
    assert {r["column"] for r in rows} == {"x", "y", "name", "flag", "when", "level"}
    _, rows = view(view="correlation", method="spearman")
    assert [r["column"] for r in rows] == ["x", "y"] and rows[0]["x"] == 1.0
    _, rows = view(view="profile")
    assert len({frozenset(r) for r in rows}) == 1  # every row has every column of the report
    profile = {r["column"]: r for r in rows}
    assert profile["x"]["min"] == -40.0 and profile["name"]["mean"] is None and profile["level"]["top"] == "high"


def test_drop_rows_by_condition_and_range(client):  # noqa: F811
    alice = new_user()
    version = upload(client, alice, body=_messy_csv(), name="messy.csv")["dataset_version_id"]

    gt = _ok(_run(client, alice, version, "drop_rows", mode="condition", column="x", operator="greater_than", value="10"))
    assert _preview(client, alice, gt["active_dataset_version_id"])["x"].max() == 3.0
    kept = _ok(_run(client, alice, version, "drop_rows", mode="condition", column="name",
                    operator="contains", value="BETA", keep_matching=True))
    assert len(_preview(client, alice, kept["active_dataset_version_id"])) == 4
    missing = _ok(_run(client, alice, version, "drop_rows", mode="condition", column="name", operator="is_missing"))
    assert len(_preview(client, alice, missing["active_dataset_version_id"])) == 7
    ranged = _ok(_run(client, alice, version, "drop_rows", mode="row_range", start_row=2, end_row=5))
    assert _preview(client, alice, ranged["active_dataset_version_id"])["x"].tolist() == [1.0, 2.0, -40.0, 2.5, 2.0]

    assert "must be a number" in _run(client, alice, version, "drop_rows", mode="condition", column="x",
                                  operator="less_than", value="abc").json()["detail"]
    assert "all 8 rows" in _run(client, alice, version, "drop_rows", mode="row_range").json()["detail"]
    assert _run(client, alice, version, "drop_rows", mode="condition", column="x").status_code == 422


def test_drop_duplicates_keep_options(client):  # noqa: F811
    alice = new_user()
    version = upload(client, alice, body=_messy_csv(), name="messy.csv")["dataset_version_id"]
    first = _ok(_run(client, alice, version, "drop_duplicates"))
    assert len(_preview(client, alice, first["active_dataset_version_id"])) == 6
    by_name = _ok(_run(client, alice, version, "drop_duplicates", columns=["name"], keep="none"))
    assert "beta" not in _preview(client, alice, by_name["active_dataset_version_id"])["name"].tolist()
    last = _ok(_run(client, alice, version, "drop_duplicates", columns=["level"], keep="last"))
    assert _preview(client, alice, last["active_dataset_version_id"])["x"].tolist() == [3.0, -40.0, 2.0]


def test_convert_types_and_parse_dates(client):  # noqa: F811
    alice = new_user()
    version = upload(client, alice, body=_messy_csv(), name="messy.csv")["dataset_version_id"]

    flags = _ok(_run(client, alice, version, "convert_types", columns=["flag"], target_type="boolean"))
    assert "flag (0)" not in flags["message"]  # "?" is a placeholder for missing, not a failure
    assert _preview(client, alice, flags["active_dataset_version_id"])["flag"].tolist() == \
        [True, False, False, True, False, True, None, False]

    dates = _ok(_run(client, alice, version, "convert_types", columns=["when"], target_type="datetime"))
    assert "when (1)" in dates["message"]
    schema = client.get(f"/datasets/versions/{dates['active_dataset_version_id']}/schema", headers=as_user(alice)).json()
    assert next(c["dtype"] for c in schema["columns"] if c["name"] == "when").startswith("timestamp")

    strict = _run(client, alice, version, "convert_types", columns=["when"], target_type="datetime", errors="raise")
    assert strict.status_code == 400 and "when" in strict.json()["detail"]
    fmt = _ok(_run(client, alice, version, "convert_types", columns=["when"], target_type="datetime",
                   datetime_format="%Y-%m-%d"))
    assert "when (2)" in fmt["message"]

    fractional = _run(client, alice, version, "convert_types", columns=["x"], target_type="integer")
    assert fractional.status_code == 400 and "float" in fractional.json()["detail"]
    as_text = _ok(_run(client, alice, version, "convert_types", columns=["x"], target_type="string"))
    schema = client.get(f"/datasets/versions/{as_text['active_dataset_version_id']}/schema", headers=as_user(alice)).json()
    assert not next(c["numeric"] for c in schema["columns"] if c["name"] == "x")


def test_clean_text_operations(client):  # noqa: F811
    alice = new_user()
    version = upload(client, alice, body=_messy_csv(), name="messy.csv")["dataset_version_id"]

    stripped = _ok(_run(client, alice, version, "clean_text", columns=["name"], operation="strip"))
    lowered = _ok(_run(client, alice, stripped["active_dataset_version_id"], "clean_text",
                       columns=["name"], operation="lower"))
    names = _preview(client, alice, lowered["active_dataset_version_id"])["name"].tolist()
    assert names[0] == "alpha" and names[3] == "gamma  ray" and names[6] == "beta"
    collapsed = _ok(_run(client, alice, lowered["active_dataset_version_id"], "clean_text",
                         columns=["name"], operation="collapse_whitespace"))
    assert _preview(client, alice, collapsed["active_dataset_version_id"])["name"][3] == "gamma ray"
    replaced = _ok(_run(client, alice, version, "clean_text", columns=["level"], operation="replace",
                        find="^(l|m)\\w+", replace_with="not high", use_regex=True))
    assert set(_preview(client, alice, replaced["active_dataset_version_id"])["level"]) == {"high", "not high"}

    assert "Not text columns" in _run(client, alice, version, "clean_text", columns=["x"], operation="lower").json()["detail"]
    assert _run(client, alice, version, "clean_text", columns=["name"], operation="replace", find="(",
                use_regex=True).status_code == 422


def test_outliers_can_be_capped_or_clipped(client):  # noqa: F811
    alice = new_user()
    version = upload(client, alice, body=_messy_csv(), name="messy.csv")["dataset_version_id"]
    capped = _ok(_run(client, alice, version, "remove_outliers", columns=["x"], method="iqr", action="cap"))
    x = _preview(client, alice, capped["active_dataset_version_id"])["x"]
    assert len(x) == 8 and x.max() < 50 and x.min() > -40 and "Capped 2" in capped["message"]
    clipped = _ok(_run(client, alice, version, "remove_outliers", columns=["x"], method="clip",
                       lower_bound=0, upper_bound=10))
    assert _preview(client, alice, clipped["active_dataset_version_id"])["x"].tolist() == \
        [1.0, 2.0, 2.0, 10.0, 3.0, 0.0, 2.5, 2.0]
    assert _run(client, alice, version, "remove_outliers", columns=["x"], method="clip").status_code == 422


def test_label_and_ordered_encoding(client):  # noqa: F811
    alice = new_user()
    version = upload(client, alice, body=_messy_csv(), name="messy.csv")["dataset_version_id"]
    label = _ok(_run(client, alice, version, "encode_categorical", columns=["level"], strategy="label"))
    assert _preview(client, alice, label["active_dataset_version_id"])["level"].tolist()[:4] == [1.0, 0.0, 0.0, 2.0]
    ordered = _ok(_run(client, alice, version, "encode_categorical", columns=["level"], strategy="ordinal",
                       category_order=["low", "medium", "high"]))
    assert _preview(client, alice, ordered["active_dataset_version_id"])["level"].tolist()[:4] == [1.0, 3.0, 3.0, 2.0]
    incomplete = _run(client, alice, version, "encode_categorical", columns=["level"], strategy="ordinal",
                      category_order=["low", "high"])
    assert incomplete.status_code == 400 and "medium" in incomplete.json()["detail"]


@pytest.mark.parametrize("score_func", ["f_regression", "mutual_info_regression"])
def test_select_k_best_keeps_the_informative_features(client, score_func):  # noqa: F811
    rng = np.random.default_rng(3)
    n = 120
    df = pd.DataFrame({f"noise{i}": rng.normal(size=n) for i in range(4)})
    df["a"], df["b"] = rng.normal(size=n), rng.normal(size=n)
    df["y"] = 3 * df["a"] - 2 * df["b"] + rng.normal(scale=0.1, size=n)
    alice = new_user()
    version = upload(client, alice, body=df.to_csv(index=False).encode(), name="kbest.csv")["dataset_version_id"]
    body = _ok(_run(client, alice, version, "select_k_best", target_column="y", k=2, score_func=score_func))
    assert sorted(_preview(client, alice, body["active_dataset_version_id"]).columns) == ["a", "b", "y"]


def test_stratified_split_keeps_strata_proportions():
    X = pd.DataFrame({"a": range(100)})
    y = pd.Series(np.r_[np.zeros(80), np.ones(20)])
    args = TrainSoftSensorArgs(dataset_version_id=_VID, algorithm="OLS", target_column="y",
                               feature_columns=["a"], split_strategy="stratified", test_size=0.25)
    strata = y.astype(int).astype(str)
    _, X_test, _, y_test = _split(X, y, args, strata)
    assert len(X_test) == 25 and int(y_test.sum()) == 5
    binned = _strata(pd.Series(np.arange(100.0)), 4)
    assert binned.value_counts().tolist() == [25, 25, 25, 25]
    with pytest.raises(ValueError, match="at least 2 rows"):
        _split(X, y, args, pd.Series(["a"] * 99 + ["b"]))


@pytest.mark.parametrize(("alias", "expected"), [
    ("chronological", "sequential"), ("time-series", "sequential"), ("Stratified", "stratified"),
])
def test_split_aliases(alias, expected):
    args = TrainSoftSensorArgs(dataset_version_id=_VID, algorithm="OLS", target_column="y",
                               feature_columns=["a"], split_strategy=alias)
    assert args.split_strategy == expected


@pytest.mark.parametrize(("cv", "folds", "splits"), [
    ("kfold", 4, 4), ("stratified_kfold", 3, 3), ("time_series", 3, 3), ("leave_one_out", 5, 30), ("LOOCV", 5, 30),
])
def test_cross_validation_strategies(cv, folds, splits):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=30), "b": rng.normal(size=30)})
    y = pd.Series(2 * X["a"] - X["b"] + rng.normal(scale=0.05, size=30))
    args = TrainSoftSensorArgs(dataset_version_id=_VID, algorithm="OLS", target_column="y",
                               feature_columns=["a", "b"], cv_strategy=cv, cv_folds=folds, n_strata=3)
    result = _cross_validate(args, X, y, None)
    assert result["n_splits"] == splits and result["r2"] > 0.98
    if cv == "time_series":
        assert result["n_rows"] < 30  # the first block is never validated
    else:
        assert result["n_rows"] == 30
    if splits > 20:
        assert result["folds"] is None and result["fold_r2_mean"] is None
    else:
        assert len(result["folds"]) == splits and result["fold_rmse_std"] is not None


def test_cross_validation_limits():
    X = pd.DataFrame({"a": range(10)})
    y = pd.Series(range(10), dtype=float)
    kfold = TrainSoftSensorArgs(dataset_version_id=_VID, algorithm="OLS", target_column="y",
                                feature_columns=["a"], cv_strategy="kfold", cv_folds=11)
    with pytest.raises(ValueError, match="cv_folds"):
        _cross_validate(kfold, X, y, None)
    strat = TrainSoftSensorArgs(dataset_version_id=_VID, algorithm="OLS", target_column="y",
                                feature_columns=["a"], cv_strategy="stratified_kfold", cv_folds=5, n_strata=5)
    with pytest.raises(ValueError, match="every stratum"):
        _cross_validate(strat, X, y, None)
    assert _cross_validate(kfold.model_copy(update={"cv_strategy": "none"}), X, y, None) is None


@pytest.mark.parametrize("split", ["random", "stratified", "sequential"])
@pytest.mark.parametrize("cv", ["none", "kfold", "stratified_kfold", "leave_one_out", "time_series"])
def test_every_split_and_cv_trains_from_the_form(client, split, cv):  # noqa: F811
    alice = new_user()
    version = upload(client, alice, body=_rich_csv(), name="rich.csv")["dataset_version_id"]
    args = {**_form_arguments(_SPECS["train_soft_sensor"], "OLS"), "split_strategy": split, "cv_strategy": cv}
    fields = {f["name"]: f for f in _SPECS["train_soft_sensor"]["fields"]}
    args.update({f: fields[f]["default"] for f in ("cv_folds", "cv_shuffle", "n_strata", "random_state")
                 if _visible(fields[f], args)})
    metrics = _ok(_run(client, alice, version, "train_soft_sensor", **args))["model_metrics"]
    assert metrics["hyperparameters"]["split_strategy"] == split
    if cv == "none":
        assert metrics["cv"] is None and "cv_strategy" not in metrics["hyperparameters"]
    else:
        assert metrics["cv"]["strategy"] == cv and metrics["hyperparameters"]["cv_strategy"] == cv
        assert metrics["cv"]["rmse"] is not None
    if "stratified" in (split, cv.split("_")[0]):
        assert metrics["hyperparameters"]["stratify_column"] == "y"


def test_stratify_by_another_column(client):  # noqa: F811
    alice = new_user()
    version = upload(client, alice, body=_rich_csv(), name="rich.csv")["dataset_version_id"]
    metrics = _ok(_run(client, alice, version, "train_soft_sensor", algorithm="OLS", target_column="y",
                       feature_columns=["x1", "x2"], split_strategy="stratified", stratify_column="cat",
                       test_size=0.3))["model_metrics"]
    assert metrics["hyperparameters"]["stratify_column"] == "cat"
    missing = _run(client, alice, version, "train_soft_sensor", algorithm="OLS", target_column="y",
                   feature_columns=["x1", "x2"], split_strategy="stratified", stratify_column="nope")
    assert missing.status_code == 400 and "nope" in missing.json()["detail"]


@pytest.mark.parametrize("text", [
    "show the head of the data", "show the tail", "what is the shape", "drop duplicate rows",
    "convert ts to datetime", "strip spaces in cat", "cap the outliers", "run 5-fold cross validation",
    "use a stratified split", "selectkbest with k=3", "value counts of cat", "LOOCV please",
])
def test_router_sends_new_operations_to_the_tools(text):
    from backend.agents.router import _rule_based_intent
    assert _rule_based_intent(text) == "EXECUTE"


def test_manual_runs_are_rate_limited(client, monkeypatch):  # noqa: F811
    monkeypatch.setenv("MANUAL_RATE_LIMIT_PER_HOUR", "1")
    manual_limiter.reset()
    alice = new_user()
    version = upload(client, alice, body=_csv())["dataset_version_id"]
    assert _run(client, alice, version, "get_column_statistics").status_code == 200
    assert _run(client, alice, version, "get_column_statistics").status_code == 429
    manual_limiter.reset()
