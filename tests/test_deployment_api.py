"""Deployment step end to end: deploy a trained model, stream raw rows in, get the training-time predictions back."""

from __future__ import annotations

import io
import json
import uuid
import zipfile

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import func, select

from backend.api import deployments as deployments_api
from backend.database import AsyncSessionLocal
from backend.deployment import service
from backend.mcp_server.modeling_tools import _load_model_artifact
from backend.mcp_server.eda_tools import _load_dataframe
from backend.models import DatasetVersion, Deployment, DeploymentRecord, TrainedModel
from backend.services.projects import evict_old_datasets
from backend.auth import User
from test_deployment_replay import FEATURES, build_pipeline, plant_csv, run_tool
from test_online_api import as_user, client, new_user, storage, upload  # noqa: F401 — fixtures

MODEL_FEATURES = [f for f in FEATURES if f != "pca_0"] + ["pca_0"]


@pytest.fixture(autouse=True)
def _fresh_limits():
    deployments_api.live_limiter.reset()
    service.clear_caches()
    yield
    service.clear_caches()


def train(client, user, version):  # noqa: F811
    out = run_tool(client, user, version, "train_soft_sensor", algorithm="RIDGE", target_column="quality",
                   feature_columns=MODEL_FEATURES, test_size=0.2)
    return out["model_metrics"]["model_id"]


def deploy(client, user, version, model_id, **extra):  # noqa: F811
    resp = client.post("/deployments", headers=as_user(user),
                       json={"dataset_version_id": version, "model_id": model_id, **extra})
    assert resp.status_code == 200, resp.text
    return resp.json()


def raw_rows(body: bytes) -> list[dict]:
    frame = pd.read_csv(io.BytesIO(body))
    rows = frame[["time", "feed", "temp", "pressure", "grade"]].to_dict(orient="records")
    return [{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()} for r in rows]


def expected_predictions(client, model_id):  # noqa: F811
    async def go():
        async with AsyncSessionLocal() as session:
            row = await session.get(TrainedModel, uuid.UUID(model_id))
            version = await session.get(DatasetVersion, uuid.UUID(row.meta["dataset_version_id"]))
            return row.artifact_key, _load_dataframe(version)

    key, training = client.portal.call(go)
    artifact = _load_model_artifact(key)
    complete = training.dropna(subset=MODEL_FEATURES)
    y = artifact["model"].predict(complete[MODEL_FEATURES])
    times = pd.to_datetime(complete["time"]).dt.tz_localize("UTC")
    return dict(zip([t.isoformat() for t in times], y)), training


def test_live_rows_get_the_training_predictions(client):  # noqa: F811
    alice = new_user()
    h = as_user(alice)
    version = build_pipeline(client, alice)
    model_id = train(client, alice, version)

    plan = client.get(f"/datasets/versions/{version}/deployment-plan", params={"model_id": model_id}, headers=h)
    assert plan.status_code == 200, plan.text
    plan = plan.json()
    assert plan["ready"] is True and plan["blocking"] == []
    assert sorted(i["name"] for i in plan["inputs"]) == ["feed", "grade", "pressure", "temp", "time"]
    assert plan["model"]["model_id"] == model_id

    created = deploy(client, alice, version, model_id, name="Column quality", bias_gain=0.5)
    key, dep = created["api_key"], created["deployment"]
    assert key.startswith("ssk_") and dep["api_key_prefix"] == key[:12] and dep["name"] == "Column quality"
    live = {"X-API-Key": key}

    info = client.get(f"/live/{dep['id']}", headers=live)
    assert info.status_code == 200 and info.json()["status"] == "active"
    # The shutdown rows that step 2 filters out are not part of the training range.
    feed = next(i for i in dep["inputs"] if i["name"] == "feed")["stats"]
    running = pd.read_csv(io.BytesIO(plant_csv()))["feed"]
    running = running[running >= 10]
    assert feed["low"] == pytest.approx(running.min()) and feed["typical"] == pytest.approx(running.median())

    rows = raw_rows(plant_csv())
    records = []
    for start in range(0, len(rows), 37):
        resp = client.post(f"/live/{dep['id']}/predict", headers=live, json={"rows": rows[start:start + 37]})
        assert resp.status_code == 200, resp.text
        records.extend(resp.json()["predictions"])
    assert len(records) == len(rows)

    expected, training = expected_predictions(client, model_id)
    by_time = {r["time"]: r for r in records}
    compared = 0
    for t, y in expected.items():
        rec = by_time[t]
        assert rec["status"] in ("ok", "warning"), rec
        assert rec["prediction"] == pytest.approx(y, rel=1e-9, abs=1e-9), (t, rec)
        compared += 1
    assert compared >= 500

    shutdown = [r for r in records if r["inputs"]["feed"] is not None and r["inputs"]["feed"] < 10]
    assert len(shutdown) == 15
    assert all(r["status"] == "suppressed" and "Removed by step 2" in r["reasons"][0] for r in shutdown)
    first = records[0]
    assert first["status"] == "suppressed" and "history" in first["reasons"][0]

    # A connector resending rows after a reconnect does not create duplicates.
    again = client.post(f"/live/{dep['id']}/predict", headers=live, json={"rows": rows[-5:]}).json()["predictions"]
    assert [r["status"] for r in again] == ["ignored"] * 5

    # Lab results correct the bias: bias ← (1 − g)·bias + g·(lab − prediction).
    t_last = max(expected)
    lab_value = expected[t_last] + 2.0
    future = (pd.Timestamp(t_last) + pd.Timedelta(hours=1)).isoformat()
    # A sample taken while the plant was down: its prediction was suppressed, so the
    # normal-operation prediction a minute or two earlier must not be used instead.
    lab = client.post(f"/live/{dep['id']}/lab", headers=live, json={"samples": [
        {"time": future, "value": 1.0}, {"time": t_last, "value": lab_value}, {"time": "not a time", "value": 1},
        {"time": t_last, "value": 99.0}, {"time": shutdown[1]["time"], "value": 20.0},
    ]})
    assert lab.status_code == 200, lab.text
    results = lab.json()["results"]
    assert [r["status"] for r in results] == ["early", "matched", "rejected", "duplicate", "unmatched"]  # in request order
    assert "suppressed" in results[4]["reason"]
    assert lab.json()["bias"] == pytest.approx(0.5 * 2.0)
    dup = client.post(f"/live/{dep['id']}/lab", headers=live, json={"samples": [{"time": t_last, "value": 1.0}]})
    assert dup.json()["results"][0]["status"] == "duplicate"

    nxt = (pd.Timestamp(rows[-1]["time"]) + pd.Timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")
    newer = dict(rows[-1], time=nxt)
    rec = client.post(f"/live/{dep['id']}/predict", headers=live, json={"rows": [newer]}).json()["predictions"][0]
    assert rec["corrected"] == pytest.approx(rec["prediction"] + 1.0)

    mon = client.get(f"/deployments/{dep['id']}/monitor", headers=h, params={"limit": 50})
    assert mon.status_code == 200
    body = mon.json()
    assert len(body["records"]) == 50 and body["records"][-1]["time"] > body["records"][0]["time"]
    matched = [lab for lab in body["labs"] if lab["prediction"] is not None]
    assert len(body["labs"]) == 2 and len(matched) == 1 and matched[0]["value"] == pytest.approx(lab_value)
    assert body["accuracy"]["n"] == 1
    assert body["deployment"]["counts"]["ignored"] == 5 and body["stored_records"] == len(rows) + 1


def test_live_access_is_by_key_or_owner_and_respects_pause(client):  # noqa: F811
    alice, bob = new_user(), new_user()
    version = build_pipeline(client, alice)
    model_id = train(client, alice, version)
    created = deploy(client, alice, version, model_id)
    dep_id, key = created["deployment"]["id"], created["api_key"]
    row = {"rows": raw_rows(plant_csv())[:3]}

    assert client.post(f"/live/{dep_id}/predict", headers={"X-API-Key": "ssk_wrong"}, json=row).status_code == 401
    assert client.post(f"/live/{uuid.uuid4()}/predict", headers={"X-API-Key": key}, json=row).status_code == 401
    assert client.post(f"/live/{dep_id}/predict", headers=as_user(bob), json=row).status_code == 404
    assert client.post(f"/live/{dep_id}/predict", json=row).status_code == 401
    assert client.post(f"/live/{dep_id}/predict", headers=as_user(alice), json=row).status_code == 200  # the simulator
    assert client.post(f"/live/{dep_id}/predict", headers={"X-API-Key": key}, json={"rows": []}).status_code == 422

    for url in (f"/deployments/{dep_id}", f"/deployments/{dep_id}/monitor", f"/deployments/{dep_id}/simulation"):
        assert client.get(url, headers=as_user(bob)).status_code == 404, url
    assert client.delete(f"/deployments/{dep_id}", headers=as_user(bob)).status_code == 404

    paused = client.patch(f"/deployments/{dep_id}", headers=as_user(alice), json={"status": "paused"})
    assert paused.status_code == 200 and paused.json()["status"] == "paused"
    assert client.post(f"/live/{dep_id}/predict", headers={"X-API-Key": key}, json=row).status_code == 409
    client.patch(f"/deployments/{dep_id}", headers=as_user(alice), json={"status": "active", "bias_gain": 0.1})

    rotated = client.post(f"/deployments/{dep_id}/rotate-key", headers=as_user(alice)).json()
    assert rotated["api_key"] != key
    assert client.get(f"/live/{dep_id}", headers={"X-API-Key": key}).status_code == 401
    assert client.get(f"/live/{dep_id}", headers={"X-API-Key": rotated["api_key"]}).status_code == 200


def test_simulation_and_connector_downloads(client):  # noqa: F811
    alice = new_user()
    h = as_user(alice)
    version = build_pipeline(client, alice)
    model_id = train(client, alice, version)
    created = deploy(client, alice, version, model_id)
    dep_id = created["deployment"]["id"]

    sim = client.get(f"/deployments/{dep_id}/simulation", headers=h, params={"start": 100, "count": 20})
    assert sim.status_code == 200
    body = sim.json()
    assert body["start"] == 100 and body["total"] == 600 and len(body["rows"]) == 20
    assert set(body["rows"][0]) == {"time", "feed", "temp", "pressure", "grade"}
    assert body["labs"][0] is not None and body["labs"][1] is None  # a lab value every fifth row

    resp = client.get(f"/deployments/{dep_id}/connector", headers=h)
    assert resp.status_code == 200 and resp.headers["content-type"] == "application/zip"
    archive = zipfile.ZipFile(io.BytesIO(resp.content))
    names = set(archive.namelist())
    assert {"softsensor_connector/softsensor_connector.py", "softsensor_connector/config.json",
            "softsensor_connector/README.md", "softsensor_connector/requirements.txt"} <= names
    config = json.loads(archive.read("softsensor_connector/config.json"))
    assert config["deployment_id"] == dep_id and set(config["inputs"]) == {"feed", "temp", "pressure", "grade"}
    assert created["api_key"] not in archive.read("softsensor_connector/config.json").decode()


def test_unready_models_are_refused_and_limits_hold(client, monkeypatch):  # noqa: F811
    alice = new_user()
    h = as_user(alice)
    v = upload(client, alice, body=plant_csv())["dataset_version_id"]
    v = run_tool(client, alice, v, "rolling_aggregate", columns=["temp"], window_type="expanding",
                 agg_func="mean")["active_dataset_version_id"]
    out = run_tool(client, alice, v, "train_soft_sensor", algorithm="RIDGE", target_column="quality",
                   feature_columns=["feed", "temp_expanding_mean"], test_size=0.2)
    model_id = out["model_metrics"]["model_id"]
    plan = client.get(f"/datasets/versions/{v}/deployment-plan", params={"model_id": model_id}, headers=h).json()
    assert plan["ready"] is False and any("expanding" in b for b in plan["blocking"])
    refused = client.post("/deployments", headers=h, json={"dataset_version_id": v, "model_id": model_id})
    assert refused.status_code == 409

    monkeypatch.setenv("MAX_DEPLOYMENTS_PER_USER", "1")
    good = run_tool(client, alice, v, "train_soft_sensor", algorithm="RIDGE", target_column="quality",
                    feature_columns=["feed", "temp"], test_size=0.2)["model_metrics"]["model_id"]
    deploy(client, alice, v, good)
    again = client.post("/deployments", headers=h, json={"dataset_version_id": v, "model_id": good})
    assert again.status_code == 409 and "at most 1" in again.json()["detail"]
    assert client.get("/deployments/versions-not-a-uuid", headers=h).status_code == 404


def test_deployed_datasets_survive_eviction_and_deletes_clean_up(client, storage):  # noqa: F811
    alice = new_user()
    h = as_user(alice)
    v = upload(client, alice, body=plant_csv())["dataset_version_id"]
    model_id = run_tool(client, alice, v, "train_soft_sensor", algorithm="RIDGE", target_column="quality",
                        feature_columns=["feed", "temp"], test_size=0.2)["model_metrics"]["model_id"]
    dep = deploy(client, alice, v, model_id)
    dep_id = dep["deployment"]["id"]
    client.post(f"/live/{dep_id}/predict", headers={"X-API-Key": dep["api_key"]},
                json={"rows": raw_rows(plant_csv())[:10]})

    async def evict():
        async with AsyncSessionLocal() as session:
            removed = await evict_old_datasets(session, User(id=alice), keep=1)
            await session.commit()
            return removed

    assert client.portal.call(evict) == 0
    assert client.get(f"/deployments/{dep_id}", headers=h).status_code == 200

    async def bundle_key():
        async with AsyncSessionLocal() as session:
            return (await session.get(Deployment, uuid.UUID(dep_id))).bundle_key

    key = client.portal.call(bundle_key)
    assert storage.path_for(key).exists()
    listed = client.get("/deployments", headers=h).json()["deployments"]
    assert [d["id"] for d in listed] == [dep_id] and listed[0]["dataset_name"] == "data.csv"

    assert client.delete("/project", headers=h).status_code == 200

    async def counts():
        async with AsyncSessionLocal() as session:
            deps = (await session.execute(select(func.count()).select_from(Deployment).where(Deployment.owner_id == alice))).scalar_one()
            recs = (await session.execute(select(func.count()).select_from(DeploymentRecord).where(
                DeploymentRecord.deployment_id == uuid.UUID(dep_id)))).scalar_one()
            return deps, recs

    assert client.portal.call(counts) == (0, 0)
    assert not storage.path_for(key).exists()
