"""Dataset upload, preview, history, and rollback API routes."""

from __future__ import annotations

import io
import uuid
from pathlib import Path
from typing import Any, BinaryIO, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from backend.agents.context import _is_numeric
from backend.auth import User, get_current_user
from backend.database import AsyncSessionLocal
from backend.mcp_server.eda_tools import _load_dataframe, _resolve_storage_path, _write_parquet
from backend.mcp_server.manipulation_tools import owner_datasets
from backend.mcp_server.model_registry import lineage_models
from backend.models import Dataset
from backend.services.lineage_service import (
    LineageService,
    build_parquet_path,
    get_dataset_version_history,
)
from backend.services.profile import cached_profile
from backend.services.projects import evict_old_datasets, owned_dataset, owned_version

router = APIRouter(tags=["datasets"])

_PREVIEW_ROWS = 50


class VersionHistoryItem(BaseModel):
    id: str
    dataset_id: str
    parent_version_id: Optional[str] = None
    file_path: str
    action_performed: str
    parameters_used: dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None


class UploadResponse(BaseModel):
    dataset_id: str
    dataset_version_id: str
    file_name: str
    row_count: int
    columns: list[str]
    table_preview: list[dict[str, Any]]
    original_schema: dict[str, Any]


class PreviewResponse(BaseModel):
    dataset_id: str
    dataset_version_id: str
    row_count: int
    columns: list[str]
    table_preview: list[dict[str, Any]]
    # Positions (0-based) of the returned rows in the version.
    row_numbers: list[int] = Field(default_factory=list)
    offset: int = 0
    sample: bool = False


class RollbackResponse(BaseModel):
    dataset_id: str
    active_dataset_version_id: str
    action_performed: str
    table_preview: list[dict[str, Any]]
    row_count: int
    columns: list[str]
    plot_data: None = None


def _df_preview(df: pd.DataFrame, limit: int = 50) -> list[dict[str, Any]]:
    preview = df.head(limit).copy()
    # JSON-safe scalars for the React table.
    return preview.astype(object).where(pd.notnull(preview), None).to_dict(orient="records")


def _read_upload_dataframe(filename: str, buffer: BinaryIO) -> pd.DataFrame:
    name = filename.lower()
    if name.endswith(".csv"):
        return pd.read_csv(buffer)
    if name.endswith(".txt"):
        return pd.read_csv(buffer, sep=None, engine="python")
    if name.endswith(".parquet") or name.endswith(".pq"):
        return pd.read_parquet(buffer)
    raise HTTPException(
        status_code=400,
        detail="Unsupported file type. Upload a .csv, .txt, or .parquet file.",
    )


def _schema_payload(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "row_count": int(len(df)),
    }


async def _protected_dataset(session: Any, version_id: Optional[str], user: User) -> Optional[uuid.UUID]:
    """The dataset of ``version_id`` when the user owns it; anything else protects nothing."""
    if not version_id:
        return None
    try:
        return (await owned_version(session, version_id, user)).dataset_id
    except HTTPException:
        return None


@router.post("/datasets/upload", response_model=UploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    keep_version_id: Optional[str] = Query(
        None, description="Active version whose dataset must survive eviction (uploading a file to combine with it)."
    ),
    user: User = Depends(get_current_user),
) -> UploadResponse:
    """Ingest CSV/Parquet, persist root Dataset + Version, return table preview."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    # Starlette spools the upload to a temporary file on disk, so it is parsed
    # from there instead of being copied into memory first.
    buffer = file.file
    buffer.seek(0, io.SEEK_END)
    if buffer.tell() == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    buffer.seek(0)

    try:
        df = await run_in_threadpool(_read_upload_dataframe, file.filename, buffer)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {exc}") from exc

    if df.empty:
        raise HTTPException(status_code=400, detail="Dataset has no rows.")

    schema = _schema_payload(df)

    async with AsyncSessionLocal() as session:
        await evict_old_datasets(session, user, protect=await _protected_dataset(session, keep_version_id, user))
        dataset = Dataset(
            owner_id=user.id,
            file_name=Path(file.filename).name,
            original_schema=schema,
        )
        session.add(dataset)
        await session.flush()

        version_id = uuid.uuid4()
        relative_path = build_parquet_path(user.id, dataset.id, version_id)
        _write_parquet(relative_path, df)

        svc = LineageService(session)
        version = await svc.save_version(
            dataset_id=dataset.id,
            action_performed="initial_upload",
            parameters_used={"source_filename": Path(file.filename).name},
            parent_version_id=None,
            file_path=relative_path,
            version_id=version_id,
        )
        await session.commit()

        return UploadResponse(
            dataset_id=str(dataset.id),
            dataset_version_id=str(version.id),
            file_name=dataset.file_name,
            row_count=int(len(df)),
            columns=list(df.columns),
            table_preview=_df_preview(df),
            original_schema=schema,
        )


@router.get("/datasets/{dataset_id}/history", response_model=list[VersionHistoryItem])
async def dataset_history(
    dataset_id: str,
    user: User = Depends(get_current_user),
) -> list[VersionHistoryItem]:
    """Return chronological version lineage for the Version History timeline."""
    async with AsyncSessionLocal() as session:
        dataset = await owned_dataset(session, dataset_id, user)
        versions = await get_dataset_version_history(session, dataset.id)
        return [
            VersionHistoryItem(
                id=str(v.id),
                dataset_id=str(v.dataset_id),
                parent_version_id=str(v.parent_version_id) if v.parent_version_id else None,
                file_path=v.file_path,
                action_performed=v.action_performed,
                parameters_used=v.parameters_used or {},
                created_at=v.created_at.isoformat() if v.created_at else None,
            )
            for v in versions
        ]


@router.get(
    "/datasets/versions/{dataset_version_id}/preview",
    response_model=PreviewResponse,
)
async def version_preview(
    dataset_version_id: str,
    limit: int = Query(25, ge=1, le=10000),
    offset: int = Query(0, ge=0, description="First row of the page."),
    sample: bool = Query(False, description="A random sample of `limit` rows (in row order) instead of a page."),
    seed: int = Query(0, description="Seed of the random sample."),
    user: User = Depends(get_current_user),
) -> PreviewResponse:
    """A page of rows (or a random sample) for the Data Preview tab, with the version's total row count."""
    async with AsyncSessionLocal() as session:
        version = await owned_version(session, dataset_version_id, user)
        try:
            df = _load_dataframe(version)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        total = int(len(df))
        if sample:
            positions = np.sort(np.random.default_rng(seed).choice(total, size=min(limit, total), replace=False))
            page, offset = df.iloc[positions], 0
        else:
            offset = min(offset, max(total - 1, 0))
            positions = np.arange(offset, min(offset + limit, total))
            page = df.iloc[offset:offset + limit]
        return PreviewResponse(
            dataset_id=str(version.dataset_id),
            dataset_version_id=str(version.id),
            row_count=total,
            columns=list(df.columns),
            table_preview=_df_preview(page, limit=limit),
            row_numbers=[int(p) for p in positions],
            offset=offset,
            sample=sample,
        )


@router.post(
    "/datasets/versions/{dataset_version_id}/rollback",
    response_model=RollbackResponse,
)
async def rollback_to_version(
    dataset_version_id: str,
    user: User = Depends(get_current_user),
) -> RollbackResponse:
    """
    Time-travel to an existing lineage node.

    Does not mutate parquet history; activates the selected version and returns
    a fresh table preview so the UI can refresh without a full page reload.
    """
    async with AsyncSessionLocal() as session:
        version = await owned_version(session, dataset_version_id, user)
        try:
            df = _load_dataframe(version)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        return RollbackResponse(
            dataset_id=str(version.dataset_id),
            active_dataset_version_id=str(version.id),
            action_performed=version.action_performed,
            table_preview=_df_preview(df),
            row_count=int(len(df)),
            columns=list(df.columns),
            plot_data=None,
        )


@router.get("/datasets/versions/{dataset_version_id}/schema")
async def version_schema(
    dataset_version_id: str,
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Columns (name, dtype, numeric), trained models and the user's other datasets, for the manual tool forms."""
    async with AsyncSessionLocal() as session:
        version = await owned_version(session, dataset_version_id, user)
        try:
            parquet = pq.ParquetFile(_resolve_storage_path(version.file_path))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        models = await lineage_models(session, str(version.id))
        datasets = [
            {
                "dataset_id": str(dataset.id),
                "file_name": dataset.file_name,
                "latest_version_id": str(latest.id),
                "current": dataset.id == version.dataset_id,
            }
            for dataset, latest in await owner_datasets(session, user.id)
        ]
        return {
            "dataset_version_id": str(version.id),
            "row_count": parquet.metadata.num_rows,
            "columns": [
                {"name": f.name, "dtype": str(f.type), "numeric": _is_numeric(f.type)}
                for f in parquet.schema_arrow
                if not f.name.startswith("__index_level")
            ],
            "models": models,
            "datasets": datasets,
        }


@router.get("/datasets/versions/{dataset_version_id}/profile")
async def version_profile(
    dataset_version_id: str,
    target: Optional[str] = Query(None, description="Target (lab) column: reported as coverage, not as missing data."),
    time_column: Optional[str] = Query(None, description="Time column; detected when omitted."),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Quality profile of the whole version: per-column checks, time sampling and gaps, target coverage."""
    async with AsyncSessionLocal() as session:
        version = await owned_version(session, dataset_version_id, user)
    try:
        profile = await run_in_threadpool(
            cached_profile,
            (str(version.id), target, time_column),
            lambda: _load_dataframe(version),
            target=target,
            time_column=time_column,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"dataset_version_id": str(version.id), **profile}


@router.get("/datasets/versions/{dataset_version_id}/meta")
async def version_meta(
    dataset_version_id: str,
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Lightweight lookup used when only a version id is known."""
    async with AsyncSessionLocal() as session:
        version = await owned_version(session, dataset_version_id, user)
        dataset = await session.get(Dataset, version.dataset_id)
        return {
            "dataset_id": str(version.dataset_id),
            "dataset_version_id": str(version.id),
            "file_name": dataset.file_name if dataset else None,
            "action_performed": version.action_performed,
        }
