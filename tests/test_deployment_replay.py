"""Deployment replay: a model's preprocessing must give the same inputs on live rows as in training.

Pipelines are built with the real tools (through the manual-run API) and then
compiled; every step is checked against the version the tool stored.
"""

from __future__ import annotations

import uuid

import numpy as np
import pandas as pd
import pytest

from backend.database import AsyncSessionLocal
from backend.deployment import replay
from backend.deployment.service import load_chain
from test_online_api import as_user, client, new_user, storage, upload  # noqa: F401 — fixtures


def plant_csv(n: int = 600, seed: int = 7) -> bytes:
    """Minute data from a column with a sparse lab target, a shutdown and a few glitches."""
    rng = np.random.default_rng(seed)
    t = pd.date_range("2024-01-01", periods=n, freq="min")
    feed = 50 + np.cumsum(rng.normal(scale=0.3, size=n))
    feed[200:215] = 2.0  # shutdown
    temp = 300 + 0.5 * feed + rng.normal(scale=1.0, size=n)
    pressure = 10 + 0.02 * temp + rng.normal(scale=0.05, size=n)
    temp[[50, 51, 400]] = np.nan
    pressure[[120, 330]] = np.nan
    grade = rng.choice(["A", "B", "C"], size=n)
    quality = 0.8 * feed - 0.1 * temp + 3 * pressure + np.where(grade == "B", 2.0, 0.0) + rng.normal(scale=0.2, size=n)
    lab = np.full(n, np.nan)
    lab[::5] = quality[::5]
    frame = pd.DataFrame({
        "time": t.strftime("%Y-%m-%d %H:%M:%S"), "feed": feed.round(4), "temp": temp.round(4),
        "pressure": pressure.round(5), "grade": grade, "quality": lab.round(4),
    })
    return frame.to_csv(index=False).encode()


def run_tool(client, user, version, tool, **arguments):  # noqa: F811
    resp = client.post(f"/api/tools/{tool}/run", headers=as_user(user),
                       json={"dataset_version_id": version, "arguments": arguments})
    assert resp.status_code == 200, resp.text
    return resp.json()


def step(client, user, version, tool, **arguments):  # noqa: F811
    out = run_tool(client, user, version, tool, **arguments)
    assert out["active_dataset_version_id"] != version, out
    return out["active_dataset_version_id"]


def build_pipeline(client, user):  # noqa: F811
    v = upload(client, user, body=plant_csv())["dataset_version_id"]
    v = step(client, user, v, "convert_types", columns=["time"], target_type="datetime")
    v = step(client, user, v, "drop_rows", mode="condition", column="feed", operator="less_than", value="10")
    v = step(client, user, v, "remove_missing_data", strategy="fill_mean", columns=["temp", "pressure"])
    v = step(client, user, v, "remove_outliers", columns=["temp"], method="z_score", threshold=3.0, action="cap")
    v = step(client, user, v, "create_lag_features", columns=["feed", "temp"], lag_periods=[1, 3])
    v = step(client, user, v, "rolling_aggregate", columns=["pressure"], window=5, agg_func="mean")
    v = step(client, user, v, "execute_formula", new_column_name="ratio", formula="feed / pressure")
    v = step(client, user, v, "normalize_data", columns=["feed", "temp", "temp_lag3", "ratio"], strategy="standard")
    v = step(client, user, v, "encode_categorical", columns=["grade"], strategy="one_hot")
    v = step(client, user, v, "reduce_dimensions", columns=["feed_lag1", "pressure_rolling5_mean"],
             strategy="pca", n_components=1, keep_original=True)
    return v


async def compile_for(version: str, features: list[str], target: str = "quality") -> replay.Compiled:
    async with AsyncSessionLocal() as session:
        chain = await load_chain(session, version)
    return replay.compile_lineage(chain, feature_columns=features, target_column=target)


FEATURES = ["feed", "temp", "temp_lag3", "feed_lag1", "pressure_rolling5_mean", "ratio", "grade_B", "grade_C", "pca_0"]


def test_plant_pipeline_replays_exactly(client):  # noqa: F811
    user = new_user()
    version = build_pipeline(client, user)
    compiled = client.portal.call(compile_for, version, FEATURES)

    assert compiled.blocking == [], compiled.blocking
    by_action = {s.action: s for s in compiled.steps}
    assert all(s.verified for s in compiled.steps if s.kept), [(s.title, s.detail) for s in compiled.steps]
    assert by_action["drop_rows"].kept and by_action["drop_rows"].mode == replay.EXACT
    assert by_action["normalize_data"].mode == replay.FITTED
    assert by_action["create_lag_features"].mode == replay.WINDOW
    assert sorted(compiled.inputs) == ["feed", "grade", "pressure", "temp", "time"]
    assert compiled.checks["training_rows"]["ok"] is True
    assert compiled.checks["live_columns"]["ok"] is True
    assert compiled.checks["streaming"]["ok"] is True, compiled.checks
    assert compiled.buffer_rows >= 3 + 5


def test_type_changes_and_steps_that_changed_nothing_still_run_live(client):  # noqa: F811
    user = new_user()
    version = build_pipeline(client, user)
    compiled = client.portal.call(compile_for, version, FEATURES)
    by_action = {s.action: s for s in compiled.steps}
    convert = by_action["convert_types"]
    assert "time" in convert.modified and convert.verified  # a type change is a change
    assert not convert.kept and convert.mode == replay.NOT_NEEDED  # the model does not use the time column
    capping = by_action["remove_outliers"]
    assert capping.kept and capping.mode == replay.FITTED  # capped nothing in training, still caps live

    live = pd.DataFrame({
        "time": [f"2024-02-01 00:{m:02d}:00" for m in range(30)],
        "feed": np.linspace(50, 52, 30), "temp": [325.0] * 28 + [np.nan, 10_000.0],
        "pressure": [16.5] * 30, "grade": ["A", "B", "C"] * 10,
    })
    frame = replay.coerce_inputs(live, compiled.input_dtypes)
    frame[replay.ROW_ID] = np.arange(len(frame))
    out = replay.run_ops(compiled.ops, frame, live=True)
    fill = next(op for i, op in compiled.ops if i == by_action["remove_missing_data"].index)
    cap = next(op for i, op in compiled.ops if i == capping.index)
    scaler = next(op for i, op in compiled.ops if i == by_action["normalize_data"].index).scalers["temp"]
    raw_temp = scaler.inverse_transform(out[["temp"]]).ravel()
    assert raw_temp[-2] == pytest.approx(min(fill.fill["temp"], cap.bounds["temp"][1]))
    assert raw_temp[-1] == pytest.approx(cap.bounds["temp"][1])
    assert out[FEATURES].iloc[-1].notna().all()


def wide_csv() -> bytes:
    frame = pd.read_csv(__import__("io").BytesIO(plant_csv(n=500, seed=11)))
    frame.loc[[10, 11, 300], "pressure"] = np.nan
    return frame.to_csv(index=False).encode()


def build_wide_pipeline(client, user):  # noqa: F811
    v = upload(client, user, body=wide_csv())["dataset_version_id"]
    v = step(client, user, v, "convert_types", columns=["time"], target_type="datetime")
    v = step(client, user, v, "mask_values", columns=["temp"], action="mask", condition_column="feed",
             operator="less_than", value="10")
    v = step(client, user, v, "remove_missing_data", strategy="knn", columns=["feed", "temp", "pressure"], n_neighbors=3)
    v = step(client, user, v, "remove_outliers", columns=["feed"], method="winsorize", threshold=0.02)
    v = step(client, user, v, "map_values", columns=["grade"], value_mapping={"A": "alpha", "B": "beta"})
    v = step(client, user, v, "apply_function", columns=["feed"], function="log")
    v = step(client, user, v, "apply_function", columns=["temp"], function="group_mean", group_by=["grade"])
    v = step(client, user, v, "transform_features", columns=["pressure"], strategy="box_cox")
    v = step(client, user, v, "transform_features", columns=["temp"], strategy="bin_equal_freq", n_bins=4)
    v = step(client, user, v, "feature_engineer", columns=["time"], strategy="datetime")
    v = step(client, user, v, "feature_engineer", columns=["feed", "temp"], strategy="ratio")
    v = step(client, user, v, "rolling_aggregate", columns=["temp"], window_type="ewm", span=6, agg_func="mean")
    v = step(client, user, v, "create_lag_features", columns=["pressure"], lag_periods=[2], operation="diff")
    v = step(client, user, v, "encode_categorical", columns=["grade"], strategy="frequency")
    v = step(client, user, v, "detect_anomalies", columns=["feed", "temp"], contamination=0.02, drop_anomalies=False)
    v = step(client, user, v, "remove_missing_data", strategy="forward_fill", columns=["pressure_boxcox"])
    return v


def test_wide_pipeline_replays_exactly(client):  # noqa: F811
    user = new_user()
    version = build_wide_pipeline(client, user)

    async def training_columns():
        async with AsyncSessionLocal() as session:
            chain = await load_chain(session, version)
        return list(chain[-1].load().columns)

    columns = client.portal.call(training_columns)
    features = [c for c in columns if c not in ("quality", "time", "grade")]
    compiled = client.portal.call(compile_for, version, features)
    report = [(s.index, s.action, s.mode, s.kept, s.verified, s.detail, s.note) for s in compiled.steps]
    assert compiled.blocking == [], (compiled.blocking, report)
    assert all(s.verified for s in compiled.steps if s.kept), report
    assert compiled.checks["training_rows"]["ok"] is True, compiled.checks
    assert compiled.checks["live_columns"]["ok"] is True, compiled.checks
    assert compiled.checks["streaming"]["ok"] is True, compiled.checks
    assert set(compiled.inputs) == {"time", "feed", "temp", "pressure", "grade"}


def test_unused_steps_are_not_needed_and_unsupported_ones_block_only_when_used(client):  # noqa: F811
    user = new_user()
    v = upload(client, user, body=plant_csv())["dataset_version_id"]
    v = step(client, user, v, "rolling_aggregate", columns=["temp"], window_type="expanding", agg_func="mean")
    v = step(client, user, v, "create_lag_features", columns=["feed"], lag_periods=[2])

    fine = client.portal.call(compile_for, v, ["feed", "feed_lag2"])
    assert fine.blocking == []
    expanding = next(s for s in fine.steps if s.action == "rolling_aggregate")
    assert expanding.mode == replay.UNSUPPORTED and not expanding.kept
    assert fine.inputs[:1] == ["feed"] and "temp" not in fine.inputs

    blocked = client.portal.call(compile_for, v, ["feed", "temp_expanding_mean"])
    assert any("expanding" in b for b in blocked.blocking), blocked.blocking


def test_resampling_moves_the_live_feed_to_the_resampled_rows(client):  # noqa: F811
    user = new_user()
    v = upload(client, user, body=plant_csv())["dataset_version_id"]
    v = step(client, user, v, "convert_types", columns=["time"], target_type="datetime")
    v = step(client, user, v, "remove_missing_data", strategy="fill_median", columns=["temp"])
    resampled = step(client, user, v, "resample_time_series", time_column="time", frequency="5min",
                     mode="downsample", aggregation="mean", columns=["feed", "temp", "pressure", "quality"])
    v = step(client, user, resampled, "create_lag_features", columns=["feed"], lag_periods=[1])

    compiled = client.portal.call(compile_for, v, ["feed", "temp", "feed_lag1"])
    by_action = {s.action: s for s in compiled.steps}
    assert by_action["resample_time_series"].mode == replay.UPSTREAM
    assert compiled.cut_version_id == resampled
    assert by_action["remove_missing_data"].mode == replay.UPSTREAM
    assert any("comes before step" in w for w in compiled.warnings)
    assert compiled.blocking == [] and compiled.checks["streaming"]["ok"] is True
    assert set(compiled.inputs) == {"feed", "temp", "time"}


def test_live_rows_are_coerced_to_training_types():
    dtypes = {"x": "float64", "n": "int64", "t": "datetime64[ns]", "g": "str"}
    frame = pd.DataFrame({"x": ["1.5", " 2 ", "bad"], "n": [1, 2, 3], "t": ["2024-01-01T00:00:00Z", "2024-01-01 00:01:00", None], "g": ["A", None, 3]})
    out = replay.coerce_inputs(frame, dtypes)
    assert out["x"].tolist()[:2] == [1.5, 2.0] and np.isnan(out["x"].iloc[2])
    assert str(out["n"].dtype) == "int64"
    assert out["t"].iloc[0] == pd.Timestamp("2024-01-01 00:00:00") and pd.isna(out["t"].iloc[2])
    assert out["g"].iloc[0] == "A" and pd.isna(out["g"].iloc[1]) and out["g"].iloc[2] == "3"


@pytest.mark.parametrize("expression,expected", [
    ("feed / pressure", {"feed", "pressure"}),
    ("`feed rate` * 2 + temp", {"feed rate", "temp"}),
    ("feed_lag1 + 1", {"feed_lag1"}),
])
def test_formula_inputs_are_found(expression, expected):
    columns = ["feed", "pressure", "feed rate", "temp", "feed_lag1", "feed_lag"]
    assert replay.referenced_columns(expression, columns) == expected


def test_unknown_version_is_rejected():
    async def go():
        async with AsyncSessionLocal() as session:
            return await load_chain(session, str(uuid.uuid4()))

    with pytest.raises(ValueError):
        import asyncio

        asyncio.run(go())
