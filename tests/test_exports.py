"""Export step: every download carries all rows, the whole pipeline and only the caller's own data."""

from __future__ import annotations

import io

import joblib
import numpy as np
import pandas as pd

from test_online_api import as_user, client, new_user, storage, upload  # noqa: F401 — fixtures


def _csv(n: int = 120_000) -> bytes:
    rng = np.random.default_rng(1)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": 2 * x1 - x2 + rng.normal(scale=0.1, size=n)}).to_csv(index=False).encode()


def _run(client, user, version, tool, **arguments):  # noqa: F811
    resp = client.post(f"/api/tools/{tool}/run", headers=as_user(user),
                       json={"dataset_version_id": version, "arguments": arguments})
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_exports_cover_the_whole_dataset_pipeline_and_model(client):  # noqa: F811
    alice, bob = new_user(), new_user()
    h = as_user(alice)
    version = upload(client, alice, body=_csv(), name="plant run.csv")["dataset_version_id"]
    dropped = _run(client, alice, version, "drop_rows", mode="condition", column="x1", operator="greater_than", value="3")
    v2 = dropped["active_dataset_version_id"]
    trained = _run(client, alice, v2, "train_soft_sensor", algorithm="RIDGE", target_column="y",
                   feature_columns=["x1", "x2"], test_size=0.2)
    model_id = trained["model_metrics"]["model_id"]
    v3 = trained["active_dataset_version_id"]
    rows = client.get(f"/datasets/versions/{v2}/preview", params={"limit": 1}, headers=h).json()["row_count"]
    assert 119_000 < rows < 120_000

    csv = client.get(f"/datasets/versions/{v2}/download", params={"format": "csv"}, headers=h)
    assert csv.status_code == 200
    assert csv.headers["content-disposition"] == f'attachment; filename="plant_run_{v2[:8]}.csv"'
    assert len(pd.read_csv(io.BytesIO(csv.content))) == rows  # streamed in batches, header once

    parquet = client.get(f"/datasets/versions/{v2}/download", params={"format": "parquet"}, headers=h)
    assert parquet.status_code == 200 and parquet.headers["content-disposition"].endswith('.parquet"')
    assert list(pd.read_parquet(io.BytesIO(parquet.content)).columns) == ["x1", "x2", "y"]

    recipe = client.get(f"/datasets/versions/{v3}/recipe", headers=h)
    assert recipe.status_code == 200 and "attachment" in recipe.headers["content-disposition"]
    body = recipe.json()
    assert body["source_file"] == "plant run.csv"
    assert [s["version_id"] for s in body["steps"]][:2] == [version, v2]
    assert body["steps"][1]["parameters"]["operator"] == "greater_than"
    assert [m["model_id"] for m in body["models"]] == [model_id]

    report = client.get(f"/datasets/versions/{v3}/report", params={"target": "y"}, headers=h)
    assert report.status_code == 200 and report.headers["content-type"].startswith("text/html")
    assert "plant run.csv" in report.text and model_id[:8] in report.text and "RIDGE" in report.text

    model = client.get(f"/models/{model_id}/download", headers=h)
    assert model.status_code == 200 and model.headers["content-disposition"].endswith('.joblib"')
    artifact = joblib.load(io.BytesIO(model.content))
    assert artifact["feature_columns"] == ["x1", "x2"] and hasattr(artifact["model"], "predict")

    other = as_user(bob)
    for url in (f"/datasets/versions/{v2}/download", f"/datasets/versions/{v3}/recipe",
                f"/datasets/versions/{v3}/report", f"/models/{model_id}/download", "/models/not-a-uuid/download"):
        assert client.get(url, headers=other).status_code == 404, url
