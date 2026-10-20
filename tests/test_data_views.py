"""The data views of the UI: paged / sampled preview with the true row count, and the quality profile."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.services.profile import profile_dataframe
from test_online_api import as_user, client, new_user, storage, upload  # noqa: F401 — fixtures


def _plant_csv(n: int = 600) -> bytes:
    rng = np.random.default_rng(0)
    t = pd.date_range("2026-01-01", periods=n, freq="10min").delete(range(300, 330))
    m = len(t)
    return pd.DataFrame({
        "time": t.astype(str),
        "temp": 300 + rng.normal(0, 1, m),
        "stuck": np.r_[rng.normal(5, 0.1, m // 2), np.full(m - m // 2, 5.0)],
        "constant": 1.0,
        "lab": np.where(np.arange(m) % 6 == 0, rng.normal(20, 1, m), np.nan),
    }).to_csv(index=False).encode()


def test_preview_pages_and_samples_with_the_true_row_count(client):  # noqa: F811
    user = new_user()
    version = upload(client, user, body=_plant_csv(), name="plant.csv")["dataset_version_id"]
    h = as_user(user)

    page = client.get(f"/datasets/versions/{version}/preview", params={"limit": 25, "offset": 100}, headers=h).json()
    assert page["row_count"] == 570
    assert page["row_numbers"] == list(range(100, 125))
    assert len(page["table_preview"]) == 25 and page["offset"] == 100 and not page["sample"]

    last = client.get(f"/datasets/versions/{version}/preview", params={"limit": 25, "offset": 560}, headers=h).json()
    assert last["row_numbers"] == list(range(560, 570))

    sample = client.get(f"/datasets/versions/{version}/preview", params={"limit": 30, "sample": True, "seed": 7}, headers=h).json()
    assert sample["sample"] and len(sample["row_numbers"]) == 30
    assert sample["row_numbers"] == sorted(sample["row_numbers"]) and max(sample["row_numbers"]) < 570
    again = client.get(f"/datasets/versions/{version}/preview", params={"limit": 30, "sample": True, "seed": 7}, headers=h).json()
    assert again["row_numbers"] == sample["row_numbers"]


def test_profile_reports_plant_data_problems_on_the_whole_version(client):  # noqa: F811
    user = new_user()
    version = upload(client, user, body=_plant_csv(), name="plant.csv")["dataset_version_id"]
    resp = client.get(f"/datasets/versions/{version}/profile", params={"target": "lab"}, headers=as_user(user))
    assert resp.status_code == 200, resp.text
    p = resp.json()
    assert p["row_count"] == 570 and p["time_column"] == "time"
    assert p["time"]["median_interval_s"] == 600 and p["time"]["n_gaps"] == 1
    assert p["time"]["gaps"][0]["duration_s"] == 31 * 600
    assert p["target"]["samples"] == 95 and p["target"]["median_interval_s"] == 3600
    cols = {c["name"]: c for c in p["columns"]}
    assert cols["lab"]["status"] == "ok" and cols["lab"]["role"] == "target"
    assert cols["constant"]["status"] == "problem"
    assert any("stuck" in i for i in cols["stuck"]["issues"])
    assert cols["temp"]["status"] == "ok"

    assert client.get(f"/datasets/versions/{version}/profile", headers=as_user(new_user())).status_code == 404


def test_profile_flags_missing_and_numbers_as_text():
    df = pd.DataFrame({
        "a": [1.0, None, None, None, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "codes": [str(i) for i in range(10)],
        "name": list("abcdefghij"),
    })
    cols = {c["name"]: c for c in profile_dataframe(df)["columns"]}
    assert cols["a"]["status"] == "warning" and cols["a"]["missing"] == 3
    assert any("stored as text" in i for i in cols["codes"]["issues"])
    assert cols["name"]["status"] == "ok"
