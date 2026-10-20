"""The EDA, manipulation, feature, validation and plotting additions, through the manual run endpoint."""

from __future__ import annotations

import base64
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from backend.agents.execution_agent import _ALL_TOOL_SCHEMAS
from backend.agents.tool_selection import CORE_TOOLS, select_schemas, select_tool_names, step_id
from backend.mcp_server.modeling_tools import TrainSoftSensorArgs, _cross_validate, _cv_folds, _holdout
from backend.mcp_server.safe_expr import assert_safe_expression
from test_online_api import as_user, client, new_user, storage, upload  # noqa: F401 — fixtures

_VID = "00000000-0000-4000-8000-000000000000"


def _run(client, user, version, tool, **arguments):  # noqa: F811
    return client.post(f"/api/tools/{tool}/run", headers=as_user(user),
                       json={"dataset_version_id": version, "arguments": arguments})


def _ok(resp):
    assert resp.status_code == 200, resp.text
    return resp.json()


def _fails(resp, *fragments, status=400):
    assert resp.status_code == status, resp.text
    for fragment in fragments:
        assert fragment in resp.json()["detail"], resp.json()["detail"]


def _preview(client, user, version):  # noqa: F811
    body = client.get(f"/datasets/versions/{version}/preview?limit=10000", headers=as_user(user)).json()
    return pd.DataFrame(body["table_preview"], columns=body["columns"])


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 80
    group = np.array(["a", "b", "c", "d"])[np.arange(n) % 4]
    x = rng.normal(size=n)
    df = pd.DataFrame({
        "g": group,
        "flag": ((np.arange(n) // 4) % 2).astype(int),
        "x": x,
        "y": 3 * x + (group == "b") * 2.0 + rng.normal(scale=0.2, size=n),
        "noise": rng.normal(size=n),
        "t": pd.date_range("2026-03-01", periods=n, freq="D").astype(str),
        "trend": np.cumsum(rng.normal(size=n)),
    })
    return df


def _setup(client, df=None, name="data.csv"):  # noqa: F811
    user = new_user()
    frame = _frame() if df is None else df
    version = upload(client, user, body=frame.to_csv(index=False).encode(), name=name)["dataset_version_id"]
    return user, version, frame


# ── Safe expressions ─────────────────────────────────────────────────────


@pytest.mark.parametrize("expr", ["a + b * 2", "`my col` > 3 and b < 2", "sqrt(abs(a)) - log(b + 1)", "a == 'x'", "~(a > 1)"])
def test_safe_expressions_are_accepted(expr):
    assert assert_safe_expression(expr) == expr


@pytest.mark.parametrize("expr", [
    "__import__('os').system('id')", "a.__class__", "@a", "a[0]", "(lambda: 1)()", "a.str.len()",
    "open('x')", "(x := 1)", "f(a=1)", "x" * 1001,
])
def test_unsafe_expressions_are_rejected(expr):
    with pytest.raises(ValueError):
        assert_safe_expression(expr)


def test_formula_and_query_refuse_code(client):  # noqa: F811
    user, version, _ = _setup(client)
    _fails(_run(client, user, version, "execute_formula", new_column_name="z", formula="x.__class__"), status=422)
    _fails(_run(client, user, version, "drop_rows", mode="query", query="__import__('os')"), status=422)
    body = _ok(_run(client, user, version, "drop_rows", mode="query", query="x > 0 and g == 'a'"))
    out = _preview(client, user, body["active_dataset_version_id"])
    df = _frame()
    assert len(out) == len(df) - int(((df["x"] > 0) & (df["g"] == "a")).sum())
    kept = _ok(_run(client, user, version, "drop_rows", mode="query", query="x > 0", keep_matching=True))
    assert (_preview(client, user, kept["active_dataset_version_id"])["x"] > 0).all()
    _fails(_run(client, user, version, "drop_rows", mode="query", query="x + 1"), "true/false")


# ── EDA ──────────────────────────────────────────────────────────────────


def test_inspect_nunique_distribution_and_alerts(client):  # noqa: F811
    df = _frame()
    df["const"] = 1.0
    df["id"] = np.arange(len(df))
    df.loc[:59, "noise"] = np.nan
    user, version, df = _setup(client, df)
    rows = _ok(_run(client, user, version, "inspect_data", view="nunique"))["table_preview"]
    assert {r["column"]: r["unique"] for r in rows}["g"] == 4

    dist = _ok(_run(client, user, version, "inspect_data", view="distribution", columns=["x"], quantiles=[0.1, 0.9]))
    row = dist["table_preview"][0]
    assert row["skew"] == pytest.approx(df["x"].skew(), abs=1e-3)
    assert row["kurtosis"] == pytest.approx(df["x"].kurtosis(), abs=1e-3)
    assert row["p10"] == pytest.approx(df["x"].quantile(0.1), abs=1e-3)
    assert row["p90"] == pytest.approx(df["x"].quantile(0.9), abs=1e-3)

    alerts = _ok(_run(client, user, version, "inspect_data", view="alerts"))["table_preview"]
    found = {(a["column"], a["alert"]) for a in alerts}
    assert ("const", "constant") in found and ("id", "id_like") in found and ("noise", "high_missing") in found


def test_aggregate_groupby_pivot_crosstab(client):  # noqa: F811
    user, version, df = _setup(client)
    rows = _ok(_run(client, user, version, "aggregate_data", group_by=["g"], value_columns=["y"], agg_func="median"))
    got = {r["g"]: r["y_median"] for r in rows["table_preview"]}
    expected = df.groupby("g")["y"].median()
    assert got == pytest.approx(expected.to_dict(), abs=1e-6)
    assert {r["g"]: r["rows"] for r in rows["table_preview"]} == {"a": 20, "b": 20, "c": 20, "d": 20}

    pivot = _ok(_run(client, user, version, "aggregate_data", mode="pivot_table", group_by=["g"],
                     pivot_column="flag", value_columns=["x"], agg_func="sum"))["table_preview"]
    want = pd.pivot_table(df, index="g", columns="flag", values="x", aggfunc="sum")
    assert pivot[1]["0"] == pytest.approx(want.loc["b", 0]) and pivot[1]["1"] == pytest.approx(want.loc["b", 1])

    shares = _ok(_run(client, user, version, "aggregate_data", mode="crosstab", group_by=["g"],
                      pivot_column="flag", normalize="index"))["table_preview"]
    assert all(r["0"] + r["1"] == pytest.approx(1.0) for r in shares)
    _fails(_run(client, user, version, "aggregate_data", mode="crosstab", group_by=["g"]), "pivot_column", status=422)


def test_statistical_tests_match_scipy(client):  # noqa: F811
    user, version, df = _setup(client)

    def test(**kw):
        return _ok(_run(client, user, version, "statistical_test", **kw))["table_preview"]

    one = test(test="t_test_one_sample", column="x", mu=0.5)[0]
    assert one["p_value"] == pytest.approx(stats.ttest_1samp(df["x"], 0.5).pvalue, abs=1e-6)
    ind = test(test="t_test_independent", column="y", group_column="flag")[0]
    a, b = df.loc[df["flag"] == 0, "y"], df.loc[df["flag"] == 1, "y"]
    assert ind["p_value"] == pytest.approx(stats.ttest_ind(a, b, equal_var=False).pvalue, abs=1e-6)
    paired = test(test="t_test_paired", column="x", column_b="noise")[0]
    assert paired["p_value"] == pytest.approx(stats.ttest_rel(df["x"], df["noise"]).pvalue, abs=1e-6)
    anova = test(test="anova", column="y", group_column="g")[0]
    assert anova["p_value"] == pytest.approx(stats.f_oneway(*[p["y"] for _, p in df.groupby("g")]).pvalue, abs=1e-6)
    chi = test(test="chi_square", column="g", column_b="flag")[0]
    assert chi["statistic"] == pytest.approx(stats.chi2_contingency(pd.crosstab(df["g"], df["flag"])).statistic, abs=1e-6)

    normal = {r["column"]: r for r in test(test="shapiro", columns=["x", "trend"])}
    assert normal["x"]["p_value"] == pytest.approx(stats.shapiro(df["x"]).pvalue, abs=1e-6)
    ks = test(test="kolmogorov_smirnov", columns=["x"])[0]
    assert 0 <= ks["p_value"] <= 1 and "normal" in ks["result"]
    adf = {r["column"]: r for r in test(test="adf", columns=["x", "trend"])}
    assert adf["x"]["result"].startswith("stationary")
    assert adf["trend"]["p_value"] > adf["x"]["p_value"]

    _fails(_run(client, user, version, "statistical_test", test="anova", column="y"), "group_column", status=422)
    _fails(_run(client, user, version, "statistical_test", test="t_test_independent", column="y", group_column="g"), "exactly 2")


# ── Data manipulation ────────────────────────────────────────────────────


def test_interpolation_window_and_lag_operations(client):  # noqa: F811
    df = pd.DataFrame({"v": [1.0, np.nan, 3.0, np.nan, np.nan, 6.0], "w": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]})
    user, version, _ = _setup(client, df)
    lin = _ok(_run(client, user, version, "remove_missing_data", strategy="interpolate", columns=["v"]))
    assert _preview(client, user, lin["active_dataset_version_id"])["v"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    _fails(_run(client, user, version, "remove_missing_data", strategy="interpolate", interpolation_method="time"),
           "time_column", status=422)

    ewm = _ok(_run(client, user, version, "rolling_aggregate", columns=["w"], window_type="ewm", span=3))
    out = _preview(client, user, ewm["active_dataset_version_id"])
    assert out["w_ewm3_mean"].tolist() == pytest.approx(df["w"].ewm(span=3).mean().tolist())
    exp = _ok(_run(client, user, version, "rolling_aggregate", columns=["w"], window_type="expanding", agg_func="max"))
    assert _preview(client, user, exp["active_dataset_version_id"])["w_expanding_max"].tolist() == df["w"].tolist()

    diff = _ok(_run(client, user, version, "create_lag_features", columns=["w"], lag_periods=[1], operation="diff"))
    assert _preview(client, user, diff["active_dataset_version_id"])["w_diff1"].tolist()[1:] == [1.0, 2.0, 4.0, 8.0, 16.0]
    lead = _ok(_run(client, user, version, "create_lag_features", columns=["w"], lag_periods=[2], operation="lead"))
    assert _preview(client, user, lead["active_dataset_version_id"])["w_lead2"].tolist()[:4] == [4.0, 8.0, 16.0, 32.0]
    pct = _ok(_run(client, user, version, "create_lag_features", columns=["w"], lag_periods=[1], operation="pct_change"))
    assert _preview(client, user, pct["active_dataset_version_id"])["w_pct_change1"].tolist()[1:] == [1.0] * 5


def test_sort_reshape_and_mapping(client):  # noqa: F811
    user, version, df = _setup(client)
    srt = _ok(_run(client, user, version, "sort_data", by=["g", "x"], descending_columns=["x"]))
    out = _preview(client, user, srt["active_dataset_version_id"])
    want = df.sort_values(["g", "x"], ascending=[True, False])
    assert out["g"].tolist() == want["g"].tolist() and out["x"].tolist() == pytest.approx(want["x"].tolist())

    long = _ok(_run(client, user, version, "reshape_data", mode="melt", id_columns=["t"], value_columns=["x", "y"]))
    melted = _preview(client, user, long["active_dataset_version_id"])
    assert list(melted.columns) == ["t", "variable", "value"] and len(melted) == 2 * len(df)
    wide = _ok(_run(client, user, long["active_dataset_version_id"], "reshape_data", mode="unstack",
                    id_columns=["t"], pivot_column="variable", value_columns=["value"]))
    back = _preview(client, user, wide["active_dataset_version_id"]).set_index("t")
    assert back["x"].tolist() == pytest.approx(df.set_index("t")["x"].tolist())

    masked = _ok(_run(client, user, version, "mask_values", columns=["x"], operator="less_than", value="0", replacement="0"))
    assert (_preview(client, user, masked["active_dataset_version_id"])["x"] >= 0).all()
    kept = _ok(_run(client, user, version, "mask_values", columns=["y"], action="where", condition_column="g",
                    operator="equals", value="a"))
    y = _preview(client, user, kept["active_dataset_version_id"])["y"]
    assert y[df["g"] != "a"].isna().all() and y[df["g"] == "a"].notna().all()

    mapped = _ok(_run(client, user, version, "map_values", columns=["g", "flag"],
                      value_mapping={"a": "first", "1": "10"}))
    out = _preview(client, user, mapped["active_dataset_version_id"])
    assert set(out["g"]) == {"first", "b", "c", "d"} and set(out["flag"]) == {0, 10}

    applied = _ok(_run(client, user, version, "apply_function", columns=["x"], function="group_mean", group_by=["g"]))
    out = _preview(client, user, applied["active_dataset_version_id"])
    new = [c for c in out.columns if c not in df.columns]
    assert len(new) == 1
    assert out[new[0]].tolist() == pytest.approx(df.groupby("g")["x"].transform("mean").tolist())
    _fails(_run(client, user, version, "apply_function", columns=["x"], function="group_sum"), "group_by", status=422)


def test_combine_datasets_is_owner_only(client):  # noqa: F811
    user, version, df = _setup(client, name="left.csv")
    extra = pd.DataFrame({"g": ["a", "b", "c"], "weight": [1.0, 2.0, 3.0]})
    upload(client, user, body=extra.to_csv(index=False).encode(), name="weights.csv")
    listed = _ok(_run(client, user, version, "list_datasets"))["table_preview"]
    assert {r["file_name"] for r in listed} == {"left.csv", "weights.csv"}

    merged = _ok(_run(client, user, version, "combine_datasets", other_dataset="weights", on=["g"], join_type="left"))
    out = _preview(client, user, merged["active_dataset_version_id"])
    assert len(out) == len(df) and out.loc[out["g"] == "d", "weight"].isna().all()
    assert out.loc[out["g"] == "b", "weight"].eq(2.0).all()
    stacked = _ok(_run(client, user, version, "combine_datasets", other_dataset="weights.csv", mode="concat",
                       join_type="outer", add_source_column=True))
    out = _preview(client, user, stacked["active_dataset_version_id"])
    assert len(out) == len(df) + 3 and set(out["source"]) == {"left.csv", "weights.csv"}

    mallory = new_user()
    theirs = upload(client, mallory, body=extra.to_csv(index=False).encode(), name="secret.csv")
    for reference in ("secret.csv", theirs["dataset_id"], theirs["dataset_version_id"]):
        _fails(_run(client, user, version, "combine_datasets", other_dataset=reference, on=["g"]), "No uploaded dataset")

    schema = client.get(f"/datasets/versions/{version}/schema", headers=as_user(user)).json()
    assert {d["file_name"]: d["current"] for d in schema["datasets"]} == {"left.csv": True, "weights.csv": False}


def test_upload_keeps_the_open_dataset(client, monkeypatch):  # noqa: F811
    monkeypatch.setenv("MAX_DATASETS_PER_USER", "2")
    user = new_user()
    first = upload(client, user, name="a.csv")
    upload(client, user, name="b.csv")
    resp = client.post(f"/datasets/upload?keep_version_id={first['dataset_version_id']}", headers=as_user(user),
                       files={"file": ("c.csv", b"a,b\n1,2\n", "text/csv")})
    assert resp.status_code == 200, resp.text
    h = as_user(user)
    assert client.get(f"/datasets/{first['dataset_id']}/history", headers=h).status_code == 200
    names = {d["file_name"] for d in client.get(f"/datasets/versions/{first['dataset_version_id']}/schema", headers=h).json()["datasets"]}
    assert names == {"a.csv", "c.csv"}
    # Someone else's version protects nothing and is not an error.
    other = upload(client, new_user())
    assert client.post(f"/datasets/upload?keep_version_id={other['dataset_version_id']}", headers=h,
                       files={"file": ("d.csv", b"a\n1\n", "text/csv")}).status_code == 200


# ── Feature engineering ──────────────────────────────────────────────────


def test_encodings_loo_and_woe(client):  # noqa: F811
    df = pd.DataFrame({"c": ["a", "a", "b", "b", "b", "c"], "y": [1.0, 3.0, 0.0, 1.0, 1.0, 1.0],
                       "bad": ["no", "yes", "no", "no", "yes", "yes"]})
    user, version, _ = _setup(client, df)
    loo = _ok(_run(client, user, version, "encode_categorical", columns=["c"], strategy="leave_one_out", target_column="y"))
    got = _preview(client, user, loo["active_dataset_version_id"])["c"].tolist()
    assert got == pytest.approx([3.0, 1.0, 1.0, 0.5, 0.5, df["y"].mean()])
    woe = _ok(_run(client, user, version, "encode_categorical", columns=["c"], strategy="woe", target_column="bad"))
    assert "Event class: yes" in woe["message"] and "Information value: c" in woe["message"]
    values = _preview(client, user, woe["active_dataset_version_id"])["c"]
    assert values[0] == values[1] and values[2] == values[3] and values[0] != values[2]
    _fails(_run(client, user, version, "encode_categorical", columns=["c"], strategy="woe", target_column="y"), "exactly two classes")


def test_reciprocal_transform(client):  # noqa: F811
    user, version, _ = _setup(client, pd.DataFrame({"a": [1.0, 2.0, 4.0, 0.0]}))
    recip = _ok(_run(client, user, version, "transform_features", columns=["a"], strategy="reciprocal"))
    r = _preview(client, user, recip["active_dataset_version_id"])["a_reciprocal"].tolist()
    assert r[:3] == [1.0, 0.5, 0.25] and (r[3] is None or math.isnan(r[3]))


def _minute_data() -> pd.DataFrame:
    """Three hours of 1-minute data, with a gap at 01:10–01:19 and a lab value every 30 min."""
    stamps = pd.date_range("2026-03-01 00:00", periods=180, freq="min")
    df = pd.DataFrame({
        "ts": stamps.astype(str),
        "temp": np.arange(180, dtype=float),
        "mode": ["run"] * 120 + ["idle"] * 60,
        "lab": [float(i) if i % 30 == 0 else np.nan for i in range(180)],
    })
    return df.drop(index=range(70, 80)).reset_index(drop=True)


def test_resample_downsamples_by_interval(client):  # noqa: F811
    user, version, df = _setup(client, _minute_data())
    hourly = _ok(_run(client, user, version, "resample_time_series", time_column="ts", frequency="1h",
                      columns=["temp", "mode"]))
    assert "Downsampled 170 rows (about 1 min apart) to 3 rows every 1h" in hourly["message"]
    out = _preview(client, user, hourly["active_dataset_version_id"])
    assert list(out.columns) == ["ts", "temp", "mode"]
    assert out["temp"].tolist() == pytest.approx([29.5, df["temp"][60:110].mean(), 149.5])
    assert out["mode"].tolist() == ["run", "run", "idle"]
    assert pd.to_datetime(out["ts"]).dt.hour.tolist() == [0, 1, 2]

    ten = _ok(_run(client, user, version, "resample_time_series", time_column="ts", frequency="10 minutes",
                   aggregation="count", columns=["temp"]))
    counts = _preview(client, user, ten["active_dataset_version_id"])["temp"].tolist()
    assert len(counts) == 17 and set(counts) == {10} and "1 empty interval(s) dropped" in ten["message"]

    kept = _ok(_run(client, user, version, "resample_time_series", time_column="ts", frequency="10min",
                    aggregation="sum", columns=["temp"], drop_empty=False))
    sums = _preview(client, user, kept["active_dataset_version_id"])["temp"].tolist()
    assert len(sums) == 18 and (sums[7] is None or math.isnan(sums[7])) and sums[0] == sum(range(10))


def test_resample_upsamples_and_fills(client):  # noqa: F811
    lab = pd.DataFrame({"ts": ["2026-03-01 00:00", "2026-03-01 01:00", "2026-03-01 03:00"], "y": [0.0, 6.0, 18.0]})
    user, version, _ = _setup(client, lab)
    linear = _ok(_run(client, user, version, "resample_time_series", time_column="ts", frequency="30min", mode="upsample"))
    assert "Upsampled 3 rows" in linear["message"] and "linear interpolation" in linear["message"]
    assert _preview(client, user, linear["active_dataset_version_id"])["y"].tolist() == pytest.approx([0, 3, 6, 9, 12, 15, 18])

    held = _ok(_run(client, user, version, "resample_time_series", time_column="ts", frequency="30min",
                    mode="upsample", fill="ffill", fill_limit=1))
    y = _preview(client, user, held["active_dataset_version_id"])["y"].tolist()
    assert y[:4] == [0.0, 0.0, 6.0, 6.0] and all(v is None or math.isnan(v) for v in y[4:6]) and y[6] == 18.0
    assert "2 numeric value(s) are missing" in held["message"]

    _fails(_run(client, user, version, "resample_time_series", time_column="ts", frequency="2h", mode="upsample"),
           "coarser", "downsampling")
    _fails(_run(client, user, version, "resample_time_series", time_column="ts", frequency="10min"),
           "finer", "upsampling")
    _fails(_run(client, user, version, "resample_time_series", time_column="ts", frequency="every so often"),
           "not understood", status=422)
    _fails(_run(client, user, version, "resample_time_series", time_column="y", frequency="1h"), "not dates")


def test_log_transform_marks_out_of_domain_values_missing(client):  # noqa: F811
    df = pd.DataFrame({"a": [0.0, math.e - 1, -1.0, -3.0]})
    user, version, _ = _setup(client, df)
    out = _ok(_run(client, user, version, "transform_features", columns=["a"], strategy="log"))
    assert "became missing: a (2)" in out["message"]
    logged = _preview(client, user, out["active_dataset_version_id"])["a_log"].tolist()
    assert logged[:2] == pytest.approx([0.0, 1.0])
    assert all(v is None or math.isnan(v) for v in logged[2:])


def test_datetime_parts_and_pairwise(client):  # noqa: F811
    df = pd.DataFrame({"d": ["2026-01-26", "2026-01-31", "2026-08-15"], "p": [1.0, 2.0, 3.0], "q": [2.0, 4.0, 5.0]})
    user, version, _ = _setup(client, df)
    parts = _ok(_run(client, user, version, "feature_engineer", columns=["d"], strategy="datetime", holiday_country="in"))
    out = _preview(client, user, parts["active_dataset_version_id"])
    assert out["d_dayofweek"].tolist() == [0, 5, 5] and out["d_is_weekend"].tolist() == [0, 1, 1]
    assert out["d_quarter"].tolist() == [1, 1, 3] and out["d_is_holiday"].tolist() == [1, 0, 1]
    _fails(_run(client, user, version, "feature_engineer", columns=["d"], strategy="datetime", holiday_country="XX"),
           status=422)
    pairs = _ok(_run(client, user, version, "feature_engineer", columns=["p", "q"], strategy="pairwise", pair_operation="ratio"))
    assert _preview(client, user, pairs["active_dataset_version_id"])["p_div_q"].tolist() == [0.5, 0.5, 0.6]


@pytest.mark.parametrize("strategy", ["truncated_svd", "lda", "tsne", "autoencoder"])
def test_new_dimensionality_reductions(client, strategy):  # noqa: F811
    user, version, df = _setup(client)
    extra = {"target_column": "g"} if strategy == "lda" else {}
    body = _ok(_run(client, user, version, "reduce_dimensions", columns=["x", "y", "noise", "trend"],
                    strategy=strategy, n_components=2, **extra))
    out = _preview(client, user, body["active_dataset_version_id"])
    new = [c for c in out.columns if c not in df.columns]
    assert len(new) == 2 and out[new].notna().all().all()


def test_model_based_selection_keeps_the_signal(client):  # noqa: F811
    user, version, _ = _setup(client)
    for method in ("tree_importance", "permutation", "shap"):
        body = _ok(_run(client, user, version, "select_features_model", target_column="y",
                        feature_columns=["x", "noise", "trend"], n_features=1, method=method))
        assert [c for c in _preview(client, user, body["active_dataset_version_id"]).columns
                if c in ("x", "noise", "trend")] == ["x"], method


@pytest.mark.parametrize("strategy", ["adasyn", "random_oversample"])
def test_oversampling_balances_classes(client, strategy):  # noqa: F811
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=50), "b": rng.normal(size=50), "label": [0] * 40 + [1] * 10})
    df.loc[40:, "a"] += 4
    user, version, _ = _setup(client, df)
    body = _ok(_run(client, user, version, "balance_data", target_column="label", strategy=strategy, k_neighbors=3))
    assert _preview(client, user, body["active_dataset_version_id"])["label"].value_counts().to_dict() == {0: 40, 1: 40}


# ── Validation strategies ────────────────────────────────────────────────


def _xy(n=60):
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series(2 * X["a"] - X["b"] + rng.normal(scale=0.1, size=n))
    groups = pd.Series([f"batch{i % 6}" for i in range(n)])
    return X, y, groups


def _args(**kw):
    return TrainSoftSensorArgs(dataset_version_id=_VID, algorithm="OLS", target_column="y", feature_columns=["a", "b"], **kw)


@pytest.mark.parametrize(("cv", "kw", "splits"), [
    ("group_kfold", {"group_column": "batch", "cv_folds": 3}, 3),
    ("leave_one_group_out", {"group_column": "batch"}, 6),
    ("repeated_kfold", {"cv_folds": 4, "cv_repeats": 3}, 12),
    ("repeated_stratified_kfold", {"cv_folds": 3, "cv_repeats": 2, "n_strata": 3}, 6),
    ("shuffle_split", {"cv_folds": 7, "cv_test_size": 0.25}, 7),
    ("stratified_shuffle_split", {"cv_folds": 5, "n_strata": 3}, 5),
    ("walk_forward", {"cv_folds": 4, "cv_window": 15}, 4),
    ("bootstrap", {"n_bootstrap": 30}, 30),
])
def test_new_cv_strategies(cv, kw, splits):
    X, y, groups = _xy()
    args = _args(cv_strategy=cv, **kw)
    folds = _cv_folds(args, X, None if "stratified" not in cv else pd.Series(pd.qcut(y, 3, labels=False)).astype(str), groups)
    assert len(folds) == splits
    for train_idx, val_idx in folds:
        assert not set(train_idx) & set(val_idx) or cv == "bootstrap"
        if cv == "bootstrap":
            assert not set(val_idx) & set(train_idx)
        if cv in ("group_kfold", "leave_one_group_out"):
            assert not set(groups.iloc[train_idx]) & set(groups.iloc[val_idx])
        if cv == "walk_forward":
            assert len(train_idx) <= 15 and train_idx.max() < val_idx.min()
    result = _cross_validate(args, X, y, None, groups)
    assert result["n_splits"] == splits and result["r2"] > 0.95


def test_holdout_and_group_split():
    X, y, groups = _xy(100)
    seq = _args(split_strategy="sequential", test_size=0.2, validation_size=0.2)
    X_train, X_val, X_test, *_ = _holdout(X, y, seq, None, None)
    assert (len(X_train), len(X_val), len(X_test)) == (60, 20, 20)
    assert X_train.index.max() < X_val.index.min() and X_val.index.max() < X_test.index.min()
    rnd = _args(test_size=0.25, validation_size=0.15)
    X_train, X_val, X_test, *_ = _holdout(X, y, rnd, None, None)
    assert len(X_train) + len(X_val) + len(X_test) == 100 and len(X_test) == 25 and 14 <= len(X_val) <= 16
    grp = _args(split_strategy="group", group_column="batch", test_size=0.34)
    X_train, _, X_test, *_ = _holdout(X, y, grp, None, groups)
    assert not set(groups[X_train.index]) & set(groups[X_test.index])
    with pytest.raises(ValueError):
        _args(test_size=0.5, validation_size=0.4)
    with pytest.raises(ValueError, match="group_column"):
        _args(cv_strategy="group_kfold")


def test_training_with_validation_set_and_groups(client):  # noqa: F811
    user, version, _ = _setup(client)
    body = _ok(_run(client, user, version, "train_soft_sensor", algorithm="RIDGE", target_column="y",
                    feature_columns=["x", "noise"], validation_size=0.2, split_strategy="group", group_column="g",
                    test_size=0.25, cv_strategy="leave_one_group_out"))
    m = body["model_metrics"]
    assert m["validation"]["n_rows"] == m["n_validation"] == 20 and m["n_test"] == 20 and m["n_train"] == 40
    assert m["hyperparameters"]["group_column"] == "g" and m["cv"]["n_splits"] == 2
    assert m["train"]["n_rows"] == 40 and 0 < m["train"]["r2"] <= 1 and m["train"]["rmse"] > 0
    assert f"train R² {m['train']['r2']:.3f}" in body["message"] and "20 test rows held out, not tested yet" in body["message"]
    assert [(s["set"], s["rows"]) for s in m["split"]] == [("train", 40), ("validation", 20), ("test", 20)]
    scored = body["active_dataset_version_id"]
    models = _ok(_run(client, user, scored, "list_trained_models"))["table_preview"]
    assert models[0]["validation_r2"] == m["validation"]["r2"] and models[0]["r2_score"] is None

    tested = _ok(_run(client, user, scored, "evaluate_on_test_set"))["model_metrics"]
    assert tested["test"]["n_rows"] == 20 and tested["validation"] == m["validation"]
    assert _ok(_run(client, user, scored, "list_trained_models"))["table_preview"][0]["r2_score"] == tested["r2_score"]

    parts = {}
    for subset in ("train", "validation", "test"):
        rows = _ok(_run(client, user, scored, "view_data_split", subset=subset))["table_preview"]
        parts[subset] = {r["row"] for r in rows}
    assert len(parts["train"]) == 40 and len(parts["validation"]) == len(parts["test"]) == 20
    assert not (parts["train"] & parts["validation"] or parts["train"] & parts["test"] or parts["validation"] & parts["test"])


@pytest.mark.parametrize("tool", ["generate_roc_curve", "generate_pr_curve", "generate_learning_curve"])
def test_model_evaluation_curves(client, tool):  # noqa: F811
    user, version, _ = _setup(client)
    trained = _ok(_run(client, user, version, "train_soft_sensor", algorithm="OLS", target_column="y",
                       feature_columns=["x", "noise"], split_strategy="stratified", n_strata=3))
    scored = trained["active_dataset_version_id"]
    extra = {"threshold": 1.0} if tool != "generate_learning_curve" else {"metric": "r2"}
    figure = _ok(_run(client, user, scored, tool, **extra))["plot_data"]
    assert figure["data"] and figure["layout"]["title"]["text"]
    if tool == "generate_roc_curve":
        assert "AUC" in figure["data"][0]["name"] and float(figure["data"][0]["name"].split("= ")[1][:-1]) > 0.8
    if tool != "generate_learning_curve":
        _fails(_run(client, user, scored, tool, threshold=1e9), "threshold")
        assert "held-out test rows" in figure["layout"]["title"]["text"]


def _points(values):
    if isinstance(values, dict):
        return np.frombuffer(base64.b64decode(values["bdata"]), dtype=values["dtype"])
    return np.asarray(values)


def test_validation_plots_use_only_the_test_rows(client):  # noqa: F811
    user, version, _ = _setup(client)
    trained = _ok(_run(client, user, version, "train_soft_sensor", algorithm="OLS", target_column="y",
                       feature_columns=["x", "noise"], test_size=0.25))
    scored = trained["active_dataset_version_id"]
    m = _ok(_run(client, user, scored, "evaluate_on_test_set"))["model_metrics"]
    parity = _ok(_run(client, user, scored, "generate_parity_plot"))["plot_data"]
    assert len(_points(parity["data"][0]["x"])) == m["n_test"]
    assert f"R² {m['r2_score']:.3f}" in parity["layout"]["title"]["text"]
    residuals = _ok(_run(client, user, scored, "generate_residuals_plot"))["plot_data"]
    assert len(_points(residuals["data"][0]["x"])) == len(_points(residuals["data"][1]["x"])) == m["n_test"]
    curve = _ok(_run(client, user, scored, "generate_learning_curve", cv_folds=5))["plot_data"]
    assert _points(curve["data"][1]["x"]).max() == m["n_train"] - math.ceil(m["n_train"] / 5)


# ── Assistant tool selection ─────────────────────────────────────────────


def test_tool_selection_keeps_the_payload_small():
    names = [s["function"]["name"] for s in _ALL_TOOL_SCHEMAS]
    assert step_id("Feature selection") == "feature-selection" and step_id("Nope") is None
    assert select_tool_names(names, user_input="anything", step_hint=None) == names
    chosen = select_tool_names(names, user_input="run an ADF test on trend", step_hint="Modeling")
    assert "statistical_test" in chosen and set(CORE_TOOLS) <= set(chosen) and "train_soft_sensor" not in chosen
    vague = select_tool_names(names, user_input="what next?", step_hint="Modeling")
    assert "train_soft_sensor" in vague and "statistical_test" not in vague
    used = select_tool_names(names, user_input="again", step_hint="Export", used=["resample_time_series"])
    assert "resample_time_series" in used
    schemas, omitted = select_schemas(_ALL_TOOL_SCHEMAS, user_input="merge with weights.csv", step_hint="Data quality")
    assert {"combine_datasets", "list_datasets"} <= {s["function"]["name"] for s in schemas}
    assert set(omitted) | {s["function"]["name"] for s in schemas} == set(names)
