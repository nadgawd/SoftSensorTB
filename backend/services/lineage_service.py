"""Dataset version lineage helpers for persistence and UI timeline rendering."""

from __future__ import annotations

import uuid
from typing import Any, Optional, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models import Dataset, DatasetVersion
from backend.storage import parquet_key


def build_parquet_path(owner_id: str, dataset_id: uuid.UUID, version_id: uuid.UUID) -> str:
    """Storage key for a dataset version's parquet file."""
    return parquet_key(owner_id, dataset_id, version_id)


async def save_dataset_version(
    session: AsyncSession,
    *,
    dataset_id: uuid.UUID,
    action_performed: str,
    parameters_used: Optional[dict[str, Any]] = None,
    parent_version_id: Optional[uuid.UUID] = None,
    file_path: Optional[str] = None,
    version_id: Optional[uuid.UUID] = None,
) -> DatasetVersion:
    """
    Persist a new dataset version and link it to its parent for lineage rollback.

    If ``file_path`` is omitted, the storage key is derived from the dataset's
    owner and the new version id.
    """
    new_id = version_id or uuid.uuid4()
    if file_path is None:
        dataset = await session.get(Dataset, dataset_id)
        if dataset is None:
            raise ValueError(f"Dataset not found: {dataset_id}")
        file_path = build_parquet_path(dataset.owner_id, dataset_id, new_id)
    resolved_path = file_path

    if parent_version_id is not None:
        parent = await session.get(DatasetVersion, parent_version_id)
        if parent is None:
            raise ValueError(f"Parent version not found: {parent_version_id}")
        if parent.dataset_id != dataset_id:
            raise ValueError(
                f"Parent version {parent_version_id} does not belong to "
                f"dataset {dataset_id}"
            )

    version = DatasetVersion(
        id=new_id,
        dataset_id=dataset_id,
        parent_version_id=parent_version_id,
        file_path=resolved_path,
        action_performed=action_performed,
        parameters_used=parameters_used or {},
    )
    session.add(version)
    await session.flush()
    await session.refresh(version)
    return version


async def get_dataset_version_history(
    session: AsyncSession,
    dataset_id: uuid.UUID,
) -> Sequence[DatasetVersion]:
    """
    Fetch the full chronological version history for a dataset.

    Ordered by ``created_at`` ascending so the UI can render a timeline from
    oldest (root upload) to newest (latest transformation).
    """
    stmt = (
        select(DatasetVersion)
        .where(DatasetVersion.dataset_id == dataset_id)
        .order_by(DatasetVersion.created_at.asc())
    )
    result = await session.execute(stmt)
    return result.scalars().all()


class LineageService:
    """Thin service wrapper around lineage helper functions."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save_version(
        self,
        *,
        dataset_id: uuid.UUID,
        action_performed: str,
        parameters_used: Optional[dict[str, Any]] = None,
        parent_version_id: Optional[uuid.UUID] = None,
        file_path: Optional[str] = None,
        version_id: Optional[uuid.UUID] = None,
    ) -> DatasetVersion:
        return await save_dataset_version(
            self._session,
            dataset_id=dataset_id,
            action_performed=action_performed,
            parameters_used=parameters_used,
            parent_version_id=parent_version_id,
            file_path=file_path,
            version_id=version_id,
        )

    async def get_history(self, dataset_id: uuid.UUID) -> Sequence[DatasetVersion]:
        return await get_dataset_version_history(self._session, dataset_id)
