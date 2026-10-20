"""Model step › Compare models: every model of the lineage with its scores, and one model's full report."""

from __future__ import annotations

import uuid

import numpy as np
import pandas as pd

from backend.database import AsyncSessionLocal
from backend.models import TrainedModel
from test_online_api import as_user, client, new_user, storage, upload  # noqa: F401 — fixtures


def _csv(n: int = 400) -> bytes:
    rng = np.random.default_rng(3)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": 2 * x1 - x2 + rng.normal(scale=0.1, size=n)}).to_csv(index=False).encode()


def _run(client, user, version, tool, **arguments):  # noqa: F811
    resp = client.post(f"/api/tools/{tool}/run", headers=as_user(user),
                       json={"dataset_version_id": version, "arguments": arguments})
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_models_are_listed_with_scores_and_reported_one_by_one(client):  # noqa: F811
    alice, bob = new_user(), new_user()
    h = as_user(alice)
    version = upload(client, alice, body=_csv())["dataset_version_id"]
    ridge = _run(client, alice, version, "train_soft_sensor", algorithm="RIDGE", target_column="y",
                 feature_columns=["x1", "x2"], test_size=0.2)["model_metrics"]["model_id"]
    pls = _run(client, alice, version, "train_soft_sensor", algorithm="PLS", target_column="y",
               feature_columns=["x1", "x2"], test_size=0.2, validation_size=0.2, n_components=1)
    pls_id, latest = pls["model_metrics"]["model_id"], pls["active_dataset_version_id"]
    _run(client, alice, latest, "evaluate_on_test_set", model_id=ridge)

    listed = client.get(f"/datasets/versions/{latest}/models", headers=h)
    assert listed.status_code == 200
    models = listed.json()["models"]
    assert [m["model_id"] for m in models] == [pls_id, ridge]  # newest first
    by_id = {m["model_id"]: m for m in models}
    assert by_id[ridge]["r2_score"] > 0.9 and by_id[pls_id]["r2_score"] is None  # only the ridge model is tested
    assert by_id[pls_id]["train"]["r2"] > 0.5 and by_id[pls_id]["validation"]["n_rows"] > 0
    assert by_id[ridge]["n_train"] + by_id[ridge]["n_test"] == 400

    report = client.get(f"/datasets/versions/{latest}/models/{ridge}", headers=h)
    assert report.status_code == 200
    body = report.json()
    assert body["model_id"] == ridge and body["algorithm"] == "RIDGE" and body["tested"] is True
    assert body["feature_columns"] == ["x1", "x2"] and body["train"]["r2"] > 0.9

    async def forget_training_scores():
        async with AsyncSessionLocal() as session:
            row = await session.get(TrainedModel, uuid.UUID(pls_id))
            row.meta = {k: v for k, v in row.meta.items() if k not in ("train", "n_train", "n_test")}
            await session.commit()

    client.portal.call(forget_training_scores)  # a model saved before the index kept them
    refilled = {m["model_id"]: m for m in client.get(f"/datasets/versions/{latest}/models", headers=h).json()["models"]}
    assert refilled[pls_id]["train"] == by_id[pls_id]["train"] and refilled[pls_id]["n_test"] == by_id[pls_id]["n_test"]

    other = as_user(bob)
    for url in (f"/datasets/versions/{latest}/models", f"/datasets/versions/{latest}/models/{ridge}"):
        assert client.get(url, headers=other).status_code == 404, url
    assert client.get(f"/datasets/versions/{latest}/models/not-a-uuid", headers=h).status_code == 404
