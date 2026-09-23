"""Per-user ownership checks and project lifetime.

A user's project is every dataset they uploaded (with its versions and trained
models) plus their saved UI state. It lasts until they press New project,
which deletes all of it from the database and from storage.
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Optional, Sequence

from fastapi import HTTPException
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.auth import User, auth_required
from backend.mcp_server.eda_tools import _validate_uuid_str
from backend.models import Dataset, DatasetVersion, ProjectState, TrainedModel
from backend.storage import get_storage


def _parse_uuid(value: str) -> uuid.UUID:
    try:
        return uuid.UUID(_validate_uuid_str(value))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def owned_dataset(session: AsyncSession, dataset_id: str, user: User) -> Dataset:
    """The dataset, or 404 when it is missing or belongs to someone else."""
    dataset = await session.get(Dataset, _parse_uuid(dataset_id))
    if dataset is None or dataset.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return dataset


async def owned_version(session: AsyncSession, dataset_version_id: str, user: User) -> DatasetVersion:
    """The version, or 404 when it is missing or belongs to someone else."""
    version = await session.get(DatasetVersion, _parse_uuid(dataset_version_id))
    if version is not None:
        dataset = await session.get(Dataset, version.dataset_id)
        if dataset is not None and dataset.owner_id == user.id:
            return version
    raise HTTPException(status_code=404, detail="Dataset version not found.")


async def _delete_datasets(session: AsyncSession, datasets: Sequence[Dataset]) -> None:
    """Delete rows and stored files; the caller commits once storage is clean."""
    ids = [d.id for d in datasets]
    if not ids:
        return
    version_keys = (
        await session.execute(select(DatasetVersion.file_path).where(DatasetVersion.dataset_id.in_(ids)))
    ).scalars().all()
    model_keys = (
        await session.execute(select(TrainedModel.artifact_key).where(TrainedModel.dataset_id.in_(ids)))
    ).scalars().all()
    # SQLite does not enforce ON DELETE CASCADE unless asked to, so children go first.
    await session.execute(delete(TrainedModel).where(TrainedModel.dataset_id.in_(ids)))
    await session.execute(
        DatasetVersion.__table__.update()
        .where(DatasetVersion.dataset_id.in_(ids))
        .values(parent_version_id=None)
    )
    await session.execute(delete(DatasetVersion).where(DatasetVersion.dataset_id.in_(ids)))
    await session.execute(delete(Dataset).where(Dataset.id.in_(ids)))
    await session.flush()
    get_storage().delete([*version_keys, *model_keys])


async def delete_project(session: AsyncSession, user: User) -> dict[str, Any]:
    """Remove everything the user owns: datasets, versions, models, UI state, files."""
    datasets = (await session.execute(select(Dataset).where(Dataset.owner_id == user.id))).scalars().all()
    await _delete_datasets(session, datasets)
    await session.execute(delete(ProjectState).where(ProjectState.owner_id == user.id))
    await session.flush()
    get_storage().delete_prefix(f"{user.id}/")
    await session.commit()
    return {"deleted_datasets": len(datasets)}


def max_datasets_per_user() -> int:
    """0 means unlimited. Online, old uploads are evicted so storage stays bounded."""
    default = 3 if auth_required() else 0
    try:
        return int(os.getenv("MAX_DATASETS_PER_USER", str(default)))
    except ValueError:
        return default


async def evict_old_datasets(session: AsyncSession, user: User, keep: Optional[int] = None) -> int:
    """Keep only the newest ``keep`` datasets of the user (before a new upload is added)."""
    keep = max_datasets_per_user() if keep is None else keep
    if keep <= 0:
        return 0
    datasets = (
        await session.execute(
            select(Dataset).where(Dataset.owner_id == user.id).order_by(Dataset.created_at.desc())
        )
    ).scalars().all()
    stale = list(datasets[keep - 1 :])
    await _delete_datasets(session, stale)
    return len(stale)


async def load_state(session: AsyncSession, user: User) -> Optional[dict[str, Any]]:
    row = await session.get(ProjectState, user.id)
    return dict(row.state) if row is not None else None


async def save_state(session: AsyncSession, user: User, state: dict[str, Any]) -> None:
    row = await session.get(ProjectState, user.id)
    if row is None:
        session.add(ProjectState(owner_id=user.id, state=state))
    else:
        row.state = state
    await session.commit()
