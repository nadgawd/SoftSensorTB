"""Dataset upload, preview, history, and rollback API routes."""

from __future__ import annotations

import io
import uuid
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from backend.auth import User, get_current_user
from backend.database import AsyncSessionLocal
from backend.limits import max_upload_bytes
from backend.mcp_server.eda_tools import _load_dataframe, _write_parquet
from backend.models import Dataset
from backend.services.lineage_service import (
    LineageService,
    build_parquet_path,
    get_dataset_version_history,
)
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


def _read_upload_dataframe(filename: str, raw: bytes) -> pd.DataFrame:
    name = filename.lower()
    buffer = io.BytesIO(raw)
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


@router.post("/datasets/upload", response_model=UploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
) -> UploadResponse:
    """Ingest CSV/Parquet, persist root Dataset + Version, return table preview."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    limit = max_upload_bytes()
    raw = await file.read(limit + 1)
    if len(raw) > limit:
        raise HTTPException(
            status_code=413,
            detail=f"File is larger than the {limit // (1024 * 1024)} MB upload limit.",
        )
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        df = _read_upload_dataframe(file.filename, raw)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {exc}") from exc

    if df.empty:
        raise HTTPException(status_code=400, detail="Dataset has no rows.")

    schema = _schema_payload(df)

    async with AsyncSessionLocal() as session:
        await evict_old_datasets(session, user)
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
    user: User = Depends(get_current_user),
) -> PreviewResponse:
    """Return the first N rows for the Data Preview tab."""
    async with AsyncSessionLocal() as session:
        version = await owned_version(session, dataset_version_id, user)
        try:
            df = _load_dataframe(version)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        return PreviewResponse(
            dataset_id=str(version.dataset_id),
            dataset_version_id=str(version.id),
            row_count=int(len(df)),
            columns=list(df.columns),
            table_preview=_df_preview(df, limit=limit),
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
