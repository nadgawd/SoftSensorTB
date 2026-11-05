"""The Split panel: the saved split of a version, a preview of a split before it is saved, and a column to plot."""

from __future__ import annotations

import uuid
from typing import Any, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, ValidationError

from backend.auth import User, get_current_user
from backend.database import AsyncSessionLocal
from backend.mcp_server.data_splits import current_split, dataset_splits, load_labels, public, time_axis
from backend.mcp_server.eda_tools import _resolve_storage_path
from backend.mcp_server.modeling_tools import SplitDataArgs, preview_split
from backend.models import DataSplit, DatasetVersion
from backend.services.projects import owned_version

router = APIRouter(tags=["splits"])


def _read_columns(version: DatasetVersion, columns: set[Optional[str]]) -> pd.DataFrame:
    """Only ``columns`` (those the version has) of every row; no columns still gives the row count."""
    try:
        parquet = pq.ParquetFile(_resolve_storage_path(version.file_path))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    wanted = [f.name for f in parquet.schema_arrow if f.name in columns]
    if not wanted:
        return pd.DataFrame(index=pd.RangeIndex(parquet.metadata.num_rows))
    return parquet.read(columns=wanted).to_pandas().reset_index(drop=True)


def _validation_message(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ()))
        msg = str(err.get("msg", "invalid value")).removeprefix("Value error, ")
        parts.append(f"{loc}: {msg}" if loc else msg)
    return "; ".join(parts) or "Invalid split settings."


@router.get("/datasets/versions/{dataset_version_id}/split")
async def version_split(
    dataset_version_id: str,
    target: Optional[str] = Query(None, description="Only splits made for this target."),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """The newest saved split that fits this version's rows, and the newest one when that no longer fits."""
    async with AsyncSessionLocal() as session:
        version = await owned_version(session, dataset_version_id, user)
        rows = await dataset_splits(session, version.dataset_id)
        columns = {c for r in rows for c in ((r.meta or {}).get("target_column"), (r.meta or {}).get("time_column"))}
        df = await run_in_threadpool(_read_columns, version, columns)
        split, newest, why = await current_split(session, version, df, target)
    return {
        "dataset_version_id": dataset_version_id,
        "split": public(split) if split else None,
        "outdated": public(newest) if newest else None,
        "outdated_reason": why,
    }


class SplitPreviewRequest(BaseModel):
    arguments: dict[str, Any] = Field(default_factory=dict, description="split_data arguments.")


@router.post("/datasets/versions/{dataset_version_id}/split/preview")
async def split_preview(
    dataset_version_id: str,
    request: SplitPreviewRequest,
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """The sets a split would make (row counts, per-set summary, what is wrong with it), without saving it."""
    async with AsyncSessionLocal() as session:
        version = await owned_version(session, dataset_version_id, user)
    arguments = {k: v for k, v in request.arguments.items() if v is not None and k != "dataset_version_id"}
    try:
        args = SplitDataArgs(dataset_version_id=dataset_version_id, **arguments)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=_validation_message(exc)) from exc
    except TypeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    columns = {args.target_column, args.time_column, args.stratify_column, args.group_column}
    df = await run_in_threadpool(_read_columns, version, columns)
    try:
        return await run_in_threadpool(preview_split, df, args)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _downsample(rows: np.ndarray, values: np.ndarray, max_points: int) -> np.ndarray:
    """At most ``max_points`` of ``rows``: the lowest and highest value of each run of rows, so spikes stay visible."""
    if len(rows) <= max_points:
        return rows
    keep: set[int] = set()
    for chunk in np.array_split(np.arange(len(rows)), max(max_points // 2, 1)):
        if len(chunk):
            part = values[chunk]
            keep.add(int(chunk[np.argmin(part)]))
            keep.add(int(chunk[np.argmax(part)]))
    return rows[np.array(sorted(keep), dtype=int)]


async def _split_labels(session: Any, version: DatasetVersion, split_id: str) -> Optional[np.ndarray]:
    try:
        row = await session.get(DataSplit, uuid.UUID(split_id))
    except ValueError:
        row = None
    if row is None or row.dataset_id != version.dataset_id:
        raise HTTPException(status_code=404, detail="Split not found.")
    try:
        return await run_in_threadpool(load_labels, row)
    except FileNotFoundError:
        return None


@router.get("/datasets/versions/{dataset_version_id}/series")
async def version_series(
    dataset_version_id: str,
    column: str = Query(..., description="Numeric column to plot."),
    time_column: Optional[str] = Query(None, description="Plot against this column; against the row number when omitted."),
    split_id: Optional[str] = Query(None, description="Also return the set of every point in this saved split."),
    target: Optional[str] = Query(None, description="Also return whether each point has this target (rows without it go to no set)."),
    max_points: int = Query(4000, ge=100, le=20000),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """One column against time (or the row number), thinned to ``max_points`` keeping each stretch's extremes."""
    async with AsyncSessionLocal() as session:
        version = await owned_version(session, dataset_version_id, user)
        labels = await _split_labels(session, version, split_id) if split_id else None
    df = await run_in_threadpool(_read_columns, version, {column, time_column, target})
    if column not in df.columns:
        raise HTTPException(status_code=400, detail=f"{column!r} is not a column of this version.")
    if time_column and time_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"{time_column!r} is not a column of this version.")
    series = df[column]
    if pd.api.types.is_bool_dtype(series):
        series = series.astype(float)
    if not pd.api.types.is_numeric_dtype(series):
        raise HTTPException(status_code=400, detail=f"{column!r} is not numeric, so it cannot be plotted.")
    values = series.to_numpy(dtype=float, na_value=np.nan)
    present = np.isfinite(values)
    kind = "index"
    x_values: Optional[np.ndarray] = None
    if time_column:
        try:
            kind, x_values = time_axis(df[time_column])
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        present &= ~np.isnat(x_values) if kind == "time" else np.isfinite(x_values)
    rows = np.flatnonzero(present)
    shown = _downsample(rows, values[rows], max_points) if len(rows) else rows
    if kind == "time":
        x = np.datetime_as_string(x_values[shown], unit="ms").tolist()
    elif kind == "number":
        x = x_values[shown].tolist()
    else:
        x = shown.tolist()
    out: dict[str, Any] = {
        "dataset_version_id": dataset_version_id,
        "column": column,
        "time_column": time_column if kind != "index" else None,
        "x_kind": kind,
        "n_rows": int(len(df)),
        "n_values": int(len(rows)),
        "sampled": bool(len(shown) < len(rows)),
        "row": shown.tolist(),
        "x": x,
        "y": values[shown].tolist(),
    }
    if labels is not None and len(labels) == len(df):
        out["set"] = labels[shown].astype(int).tolist()
    if target and target in df.columns:
        out["has_target"] = df[target].notna().to_numpy()[shown].tolist()
    return out
