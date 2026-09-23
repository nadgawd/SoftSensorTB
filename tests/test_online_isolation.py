"""Guards that keep one user's chat from reaching files or data outside their dataset."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from backend.agents.execution_agent import _inject_dataset_version
from backend.database import AsyncSessionLocal, engine, init_db
from backend.mcp_server import eda_tools
from backend.mcp_server.model_registry import resolve_model
from backend.models import Dataset, DatasetVersion, TrainedModel
from backend.storage import LocalStorage, set_storage

REPO_FILE = Path(__file__).resolve().parent.parent / "requirements.txt"


@pytest.fixture
def storage(tmp_path):
    store = LocalStorage(tmp_path / "storage")
    set_storage(store)
    yield store
    set_storage(None)


def run_db(coro_fn):
    """Run against the test database on a fresh loop, then drop its connections."""
    async def main():
        try:
            await init_db()
            return await coro_fn()
        finally:
            await engine.dispose()

    return asyncio.run(main())


@pytest.fixture
def duckdb_version(storage, monkeypatch):
    key = "u1/d1/v1.parquet"
    storage.write(key, lambda path: pd.DataFrame({"x": [1, 2, 3]}).to_parquet(path, index=False))

    async def fake_get_version(session, dataset_version_id):
        return SimpleNamespace(file_path=key)

    monkeypatch.setattr(eda_tools, "_get_version", fake_get_version)
    return str(uuid.uuid4())


def query(version, sql):
    return asyncio.run(eda_tools.run_duckdb_query(version, sql, session=object()))


def test_duckdb_queries_the_dataset(duckdb_version):
    assert query(duckdb_version, "SELECT sum(x) AS total FROM dataset")["rows"] == [{"total": 6}]


@pytest.mark.parametrize(
    "sql",
    [
        f"SELECT * FROM read_csv('{REPO_FILE}')",
        f"SELECT * FROM read_text('{REPO_FILE}')",
        "SELECT * FROM read_parquet('https://example.com/x.parquet')",
        "SELECT * FROM glob('/*')",
    ],
    ids=["read_csv", "read_text", "remote_parquet", "glob"],
)
def test_duckdb_cannot_reach_files_or_urls(duckdb_version, sql):
    with pytest.raises(Exception, match="(?i)external access|disabled|permission"):
        query(duckdb_version, sql)


def test_tools_are_pinned_to_the_active_version():
    active, other = str(uuid.uuid4()), str(uuid.uuid4())
    args = _inject_dataset_version(
        "run_duckdb_query",
        {"dataset_version_id": other, "sql_query": "SELECT 1", "session": "injected"},
        active,
    )
    assert args == {"dataset_version_id": active, "sql_query": "SELECT 1"}


def test_model_ids_from_another_dataset_are_ignored(storage):
    mine, theirs, my_version = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    my_model, their_model = uuid.uuid4(), uuid.uuid4()

    async def scenario():
        async with AsyncSessionLocal() as session:
            session.add_all([
                Dataset(id=mine, owner_id="a", file_name="a.csv", original_schema={}),
                Dataset(id=theirs, owner_id="b", file_name="b.csv", original_schema={}),
            ])
            await session.flush()
            session.add(DatasetVersion(id=my_version, dataset_id=mine, file_path="a/v.parquet",
                                       action_performed="initial_upload"))
            session.add_all([
                TrainedModel(id=my_model, dataset_id=mine, artifact_key="a/m.pkl",
                             meta={"model_id": str(my_model), "dataset_version_id": str(my_version)}),
                TrainedModel(id=their_model, dataset_id=theirs, artifact_key="b/m.pkl",
                             meta={"model_id": str(their_model)}),
            ])
            await session.commit()

            picked = await resolve_model(session, str(my_version), str(their_model))
            return picked.artifact_key

    assert run_db(scenario) == "a/m.pkl"
