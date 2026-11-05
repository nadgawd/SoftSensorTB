"""Splitting is its own step: split_data saves the sets, training and testing use them, and the equation is exported."""

from __future__ import annotations

import uuid

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sqlalchemy import func, select

from backend.database import AsyncSessionLocal
from backend.mcp_server.data_splits import SplitRange, check_labels, label_counts, row_key, time_axis
from backend.mcp_server.modeling_tools import (
    SplitDataArgs,
    _split_labels,
    equation_text,
    linear_equation,
    preview_split,
)
from backend.models import DataSplit
from test_online_api import as_user, client, new_user, storage, upload  # noqa: F401 — fixtures

VID = str(uuid.uuid4())
TRAIN, VALIDATION, TEST, UNUSED = 1, 2, 3, 0


def _frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    return pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=n, freq="h"),
        "x1": x1,
        "x2": x2,
        "batch": np.arange(n) // 10,
        "y": 1.5 + 2 * x1 - 0.5 * x2 + rng.normal(scale=0.05, size=n),
    })


def _args(**kwargs) -> SplitDataArgs:
    return SplitDataArgs(dataset_version_id=VID, target_column="y", **kwargs)


# ── Strategies ──────────────────────────────────────────────────────────────


def test_random_split_gives_each_set_its_share_and_skips_rows_without_target():
    df = _frame()
    df.loc[[3, 7], "y"] = np.nan
    labels = _split_labels(df, _args(test_size=0.2, validation_size=0.1))
    counts = label_counts(labels)
    assert counts["unused"] == 2 and (labels[[3, 7]] == UNUSED).all()
    assert counts["train"] + counts["validation"] + counts["test"] == 198
    assert abs(counts["test"] - 0.2 * 198) <= 1 and abs(counts["validation"] - 0.1 * 198) <= 1
    assert (labels == _split_labels(df, _args(test_size=0.2, validation_size=0.1))).all()
    assert not (labels == _split_labels(df, _args(test_size=0.2, validation_size=0.1, random_state=7))).all()


def test_chronological_split_follows_the_time_column_not_the_row_order():
    df = _frame().sample(frac=1, random_state=3).reset_index(drop=True)
    labels = _split_labels(df, _args(split_strategy="sequential", test_size=0.25, validation_size=0.1, time_column="time"))
    train, val, test = (df.time[labels == code] for code in (TRAIN, VALIDATION, TEST))
    assert len(test) == 50
    assert train.max() < val.min() and val.max() < test.min()


def test_group_split_keeps_every_group_in_one_set():
    df = _frame()
    labels = _split_labels(df, _args(split_strategy="group", group_column="batch", test_size=0.2))
    per_batch = pd.Series(labels).groupby(df.batch).nunique()
    assert (per_batch == 1).all()
    assert set(np.unique(labels)) == {TRAIN, TEST}


def test_stratified_split_spreads_the_target_over_the_sets():
    df = _frame(400)
    labels = _split_labels(df, _args(split_strategy="stratified", test_size=0.25, n_strata=4))
    bins = pd.qcut(df.y, 4, labels=False)
    test_share = pd.Series(labels == TEST).groupby(bins).mean()
    assert ((test_share - 0.25).abs() < 0.03).all()


# ── Custom ranges ───────────────────────────────────────────────────────────


def test_custom_ranges_by_index_and_time_later_ones_win_and_rest_fills_the_gap():
    df = _frame(100)
    ranges = [
        SplitRange(set="train", by="index", start=59, end=0),  # reversed ends are swapped
        SplitRange(set="test", by="time", start=str(df.time[80]), end=None),
        SplitRange(set="none", by="index", start=50, end=54),
    ]
    labels = _split_labels(df, _args(split_strategy="custom", time_column="time", ranges=ranges, rest="validation"))
    assert (labels[:50] == TRAIN).all() and (labels[50:55] == UNUSED).all() and (labels[55:60] == TRAIN).all()
    assert (labels[60:80] == VALIDATION).all() and (labels[80:] == TEST).all()

    without_rest = _split_labels(df, _args(split_strategy="custom", time_column="time", ranges=ranges))
    assert (without_rest[60:80] == UNUSED).all()

    preview = preview_split(df, _args(split_strategy="custom", time_column="time", ranges=ranges))
    assert preview["uncovered"] == 20 and preview["no_target"] == 0 and preview["problem"] is None
    assert preview["counts"] == {"train": 55, "validation": 0, "test": 20, "unused": 25}
    assert [s["set"] for s in preview["summary"]] == ["train", "test"]


def test_custom_time_ranges_compare_wall_clock_times():
    df = _frame(48)
    df["time"] = df.time.dt.tz_localize("Europe/Berlin").astype(str)
    kind, values = time_axis(df.time)
    assert kind == "time" and str(values[5])[:19] == "2024-01-01T05:00:00"
    ranges = [SplitRange(set="test", by="time", start="2024-01-01 05:00", end="2024-01-01 06:00")]
    labels = _split_labels(df, _args(split_strategy="custom", time_column="time", ranges=ranges, rest="train"))
    assert list(np.flatnonzero(labels == TEST)) == [5, 6]


def test_split_checks_the_sets_are_big_enough():
    with pytest.raises(ValueError, match="training set has 3"):
        check_labels(np.array([TRAIN] * 3 + [TEST] * 5))
    with pytest.raises(ValueError, match="test set has 1"):
        check_labels(np.array([TRAIN] * 10 + [TEST]))
    with pytest.raises(ValueError, match="validation set has 1 row"):
        check_labels(np.array([TRAIN] * 10 + [VALIDATION] + [TEST] * 3))
    check_labels(np.array([TRAIN] * 5 + [TEST] * 2 + [UNUSED] * 4))

    df = _frame(50)
    tiny = [SplitRange(set="test", by="index", start=0, end=0)]
    preview = preview_split(df, _args(split_strategy="custom", ranges=tiny, rest="train"))
    assert "test set has 1" in preview["problem"]


def test_split_arguments_are_checked():
    for bad in (
        {"split_strategy": "custom"},
        {"split_strategy": "custom", "ranges": [{"set": "train", "by": "time", "start": "2024-01-01"}]},
        {"test_size": 0.5, "validation_size": 0.4},
        {"split_strategy": "group"},
        {"split_strategy": "group", "group_column": "y"},
        {"split_strategy": "custom", "ranges": [{"set": "train"}], "rest": "everything"},
    ):
        with pytest.raises(ValidationError):
            _args(**bad)
    args = _args(split_strategy="Manual", ranges=[{"set": "train", "start": "", "end": 9}], rest="none")
    assert args.split_strategy == "custom" and args.rest is None and args.ranges[0].start is None


def test_row_key_follows_the_rows_not_the_columns():
    df = _frame(60)
    key = row_key(df, "y", "time")
    assert row_key(df.assign(z=df.x1 * 2, x1=df.x1 / 10), "y", "time") == key
    assert row_key(df.iloc[1:].reset_index(drop=True), "y", "time") != key
    gappy = df.copy()
    gappy.loc[4, "y"] = np.nan
    assert row_key(gappy, "y", "time") != key
    assert row_key(df.iloc[::-1].reset_index(drop=True), "y", "time") != key
    assert row_key(df, "y", None) != key


# ── Equation ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("algorithm, model", [
    ("OLS", LinearRegression()),
    ("RIDGE", Ridge(alpha=3.0)),
    ("LASSO", Lasso(alpha=0.01)),
    ("PLS", PLSRegression(n_components=2, scale=True)),
    ("PCR", make_pipeline(PCA(n_components=2), LinearRegression())),
])
def test_equation_reproduces_the_model_predictions(algorithm, model):
    rng = np.random.default_rng(4)
    X = pd.DataFrame({"a": rng.normal(50, 10, 300), "b": rng.normal(size=300), "c": rng.normal(-3, 0.01, 300)})
    y = 4 + 0.3 * X.a - 2 * X.b + 100 * X.c + rng.normal(scale=0.1, size=300)
    model.fit(X, y)
    equation = linear_equation(model, algorithm=algorithm, feature_columns=list(X.columns))
    assert equation is not None and list(equation["coefficients"]) == ["a", "b", "c"]
    by_hand = equation["intercept"] + X.to_numpy() @ np.array(list(equation["coefficients"].values()))
    np.testing.assert_allclose(by_hand, np.asarray(model.predict(X)).ravel(), rtol=1e-9, atol=1e-7)


def test_equation_has_exact_zeros_for_dropped_inputs_and_none_for_knn():
    rng = np.random.default_rng(5)
    X = pd.DataFrame({"a": rng.normal(size=200), "noise": rng.normal(size=200)})
    y = 3 * X.a + rng.normal(scale=0.01, size=200)
    lasso = Lasso(alpha=0.5).fit(X, y)
    assert linear_equation(lasso, algorithm="LASSO", feature_columns=["a", "noise"])["coefficients"]["noise"] == 0.0
    knn = KNeighborsRegressor().fit(X, y)
    assert linear_equation(knn, algorithm="KNN", feature_columns=["a", "noise"]) is None
    text = equation_text({"intercept": 1.5, "coefficients": {"a": 2.0, "b": -0.25}}, "y")
    assert text == "y = 1.5 + 2·a − 0.25·b"


# ── API: split, preview, plot, train on the split, test, export ─────────────


def _run(client, user, version, tool, expect=200, **arguments):  # noqa: F811
    resp = client.post(f"/api/tools/{tool}/run", headers=as_user(user),
                       json={"dataset_version_id": version, "arguments": arguments})
    assert resp.status_code == expect, resp.text
    return resp.json()


CUSTOM = {
    "target_column": "y",
    "split_strategy": "custom",
    "time_column": "time",
    "ranges": [{"set": "train", "by": "index", "start": 0, "end": 149}, {"set": "test", "by": "index", "start": 200}],
}


def test_split_then_train_then_test_then_export(client, storage):  # noqa: F811
    alice, bob = new_user(), new_user()
    h = as_user(alice)
    df = _frame(240)
    df.loc[[5, 11], "y"] = np.nan
    v1 = upload(client, alice, body=df.to_csv(index=False).encode())["dataset_version_id"]

    assert client.get(f"/datasets/versions/{v1}/split", params={"target": "y"}, headers=h).json()["split"] is None

    preview = client.post(f"/datasets/versions/{v1}/split/preview", headers=h, json={"arguments": CUSTOM})
    assert preview.status_code == 200, preview.text
    body = preview.json()
    assert body["counts"] == {"train": 148, "validation": 0, "test": 40, "unused": 52}
    assert body["uncovered"] == 50 and body["no_target"] == 2 and body["problem"] is None
    bad = client.post(f"/datasets/versions/{v1}/split/preview", headers=h,
                      json={"arguments": {**CUSTOM, "ranges": []}})
    assert bad.status_code == 422

    saved = _run(client, alice, v1, "split_data", **CUSTOM, rest="validation")
    assert saved["message"].startswith("Saved a custom split") and "validation 50" in saved["message"]
    split = client.get(f"/datasets/versions/{v1}/split", params={"target": "y"}, headers=h).json()["split"]
    sid = split["split_id"]
    assert split["counts"] == {"train": 148, "validation": 50, "test": 40, "unused": 2}
    assert split["strategy"] == "custom" and split["time_column"] == "time" and "row_key" not in split

    series = client.get(f"/datasets/versions/{v1}/series", headers=h,
                        params={"column": "x1", "time_column": "time", "split_id": sid, "target": "y"}).json()
    assert series["x_kind"] == "time" and len(series["row"]) == 240 and not series["sampled"]
    assert series["x"][1].startswith("2024-01-01T01:00:00")
    assert [series["set"].count(code) for code in (TRAIN, VALIDATION, TEST)] == [148, 50, 40]
    assert series["has_target"][5] is False and series["set"][5] == UNUSED
    thin = client.get(f"/datasets/versions/{v1}/series", headers=h, params={"column": "x1", "max_points": 100}).json()
    assert thin["x_kind"] == "index" and thin["sampled"] and len(thin["row"]) <= 100
    assert client.get(f"/datasets/versions/{v1}/series", headers=h, params={"column": "time"}).status_code == 400

    trained = _run(client, alice, v1, "train_soft_sensor", algorithm="OLS", target_column="y",
                   feature_columns=["x1", "x2"], split_id=sid)
    m = trained["model_metrics"]
    assert m["split_id"] == sid and m["split_strategy"] == "custom"
    assert (m["n_train"], m["n_validation"], m["n_test"]) == (148, 50, 40)
    assert m["hyperparameters"]["split_id"] == sid
    assert m["equation"]["intercept"] == pytest.approx(1.5, abs=0.05)
    assert m["equation"]["coefficients"]["x1"] == pytest.approx(2, abs=0.05)
    assert m["feature_importances"]["intercept"] == m["equation"]["intercept"]
    v2 = trained["active_dataset_version_id"]

    # The model's predictions column keeps the rows, so the split still fits the new version.
    again = _run(client, alice, v2, "train_soft_sensor", algorithm="PLS", target_column="y",
                 feature_columns=["x1", "x2"], n_components=1, split_id="latest")["model_metrics"]
    assert again["split_id"] == sid and again["n_test"] == 40
    tested = _run(client, alice, again.get("dataset_version_id") or v2, "evaluate_on_test_set",
                  model_id=m["model_id"])["model_metrics"]
    assert tested["test"]["n_rows"] == 40

    wrong_target = _run(client, alice, v1, "train_soft_sensor", expect=400, algorithm="OLS", target_column="x2",
                        feature_columns=["x1"], split_id=sid)
    assert "made for the target 'y'" in wrong_target["detail"]

    dropped = _run(client, alice, v1, "drop_rows", mode="condition", column="x1", operator="greater_than", value="1.5")
    v3 = dropped["active_dataset_version_id"]
    state = client.get(f"/datasets/versions/{v3}/split", params={"target": "y"}, headers=h).json()
    assert state["split"] is None and state["outdated"]["split_id"] == sid and "rows changed" in state["outdated_reason"]
    stale = _run(client, alice, v3, "train_soft_sensor", expect=400, algorithm="OLS", target_column="y",
                 feature_columns=["x1", "x2"], split_id=sid)
    assert "no longer fits" in stale["detail"]

    recipe = client.get(f"/datasets/versions/{v2}/recipe", params={"model_id": m["model_id"]}, headers=h).json()
    assert recipe["final_model_id"] == m["model_id"]
    assert recipe["final_equation"]["coefficients"] == m["equation"]["coefficients"]
    assert recipe["final_equation"]["text"].startswith("y = ")
    assert all(model.get("equation") for model in recipe["models"])
    report = client.get(f"/datasets/versions/{v2}/report", params={"model_id": m["model_id"]}, headers=h).text
    assert "Final model · governing equation" in report and recipe["final_equation"]["text"] in report
    assert "<td>intercept</td>" in report and "Equations of the other models" in report

    other = as_user(bob)
    assert client.get(f"/datasets/versions/{v1}/split", headers=other).status_code == 404
    assert client.post(f"/datasets/versions/{v1}/split/preview", headers=other, json={"arguments": CUSTOM}).status_code == 404
    assert client.get(f"/datasets/versions/{v1}/series", headers=other, params={"column": "x1"}).status_code == 404
    bobs = upload(client, bob, body=df.to_csv(index=False).encode())["dataset_version_id"]
    assert client.get(f"/datasets/versions/{bobs}/series", headers=other,
                      params={"column": "x1", "split_id": sid}).status_code == 404
    _run(client, bob, bobs, "train_soft_sensor", expect=400, algorithm="OLS", target_column="y",
         feature_columns=["x1", "x2"], split_id=sid)

    assert client.delete("/project", headers=h).status_code == 200

    async def splits_left():
        async with AsyncSessionLocal() as session:
            return await session.scalar(select(func.count()).select_from(DataSplit).where(DataSplit.id == uuid.UUID(sid)))

    assert client.portal.call(splits_left) == 0
    assert not (storage.root / alice).exists()


def test_a_split_without_a_time_column_ignores_a_column_that_holds_no_times(client):  # noqa: F811
    user = new_user()
    df = _frame(80).assign(label=lambda d: [f"tag-{i}" for i in range(len(d))])
    version = upload(client, user, body=df.to_csv(index=False).encode())["dataset_version_id"]
    _run(client, user, version, "split_data", target_column="y", time_column="label")
    split = client.get(f"/datasets/versions/{version}/split", headers=as_user(user)).json()["split"]
    assert split["strategy"] == "random" and split["time_column"] is None
    failed = _run(client, user, version, "split_data", expect=400, target_column="y", split_strategy="sequential",
                  time_column="label")
    assert "does not hold times" in failed["detail"]
