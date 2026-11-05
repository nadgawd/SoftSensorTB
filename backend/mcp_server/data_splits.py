"""Saved train / validation / test splits.

``split_data`` records, for every row of the version it was made on, which set
the row is in; ``train_soft_sensor`` then trains on a saved split instead of
splitting by itself. Splitting and training are separate steps, and every model
trained on one split is tested on the same rows.

A split fits another version of the same dataset when the rows are the same
ones: the same number of rows, the target missing in the same rows and, when
the split has a time column, the same time stamps. Steps that keep the rows
(scaling, new feature columns, a model's predictions) keep the split; dropping,
adding or reordering rows needs a new split.
"""

from __future__ import annotations

import hashlib
import uuid
import warnings
from datetime import datetime, timezone
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models import DataSplit, Dataset, DatasetVersion
from backend.storage import get_storage, split_key

SET_CODES = {"train": 1, "validation": 2, "test": 3}
SET_NAMES = {code: name for name, code in SET_CODES.items()}
SET_ORDER = ("train", "validation", "test")
UNUSED = 0
# Older splits of a dataset are deleted; models keep their own copy of their rows.
MAX_SPLITS_PER_DATASET = 10
_CANDIDATES = 25


class SplitRange(BaseModel):
    """Rows from ``start`` to ``end`` (both included) go to ``set``."""

    set: Literal["train", "validation", "test", "none"] = Field(
        ..., description="The set these rows go to; 'none' leaves them out of every set.",
    )
    by: Literal["index", "time"] = Field(
        default="index",
        description="'index': start and end are row numbers (0 is the first row); 'time': they are times of time_column.",
    )
    start: Optional[Union[float, str]] = Field(
        default=None, description="First row number or time of the range; omit to start at the first row.",
    )
    end: Optional[Union[float, str]] = Field(
        default=None, description="Last row number or time of the range; omit to run to the last row.",
    )

    @field_validator("start", "end", mode="before")
    @classmethod
    def _blank(cls, value: Any) -> Any:
        return None if isinstance(value, str) and not value.strip() else value


def time_axis(series: pd.Series) -> tuple[str, np.ndarray]:
    """``('number', floats)`` or ``('time', naive datetime64)``; NaN / NaT where a value is missing.

    Times with a time zone are compared on their wall-clock time, which is what
    the plots show.
    """
    if pd.api.types.is_bool_dtype(series):
        raise ValueError(f"{series.name!r} is not a time or number column")
    if pd.api.types.is_numeric_dtype(series):
        return "number", series.to_numpy(dtype=float, na_value=np.nan)
    if pd.api.types.is_datetime64_any_dtype(series):
        times = series
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            times = pd.to_datetime(series, errors="coerce", format="mixed")
            if not pd.api.types.is_datetime64_any_dtype(times):
                # Mixed UTC offsets: compare them in UTC.
                times = pd.to_datetime(series, errors="coerce", format="mixed", utc=True)
        if series.notna().any() and times.notna().sum() < 0.9 * series.notna().sum():
            raise ValueError(f"{series.name!r} does not hold times")
    if getattr(times.dt, "tz", None) is not None:
        times = times.dt.tz_localize(None)
    return "time", times.to_numpy(dtype="datetime64[ns]")


def row_key(df: pd.DataFrame, target_column: str, time_column: Optional[str]) -> str:
    """Identifies the rows a split was made on (see the module docstring)."""
    if target_column not in df.columns:
        raise ValueError(f"The target {target_column!r} is not a column of this version")
    digest = hashlib.sha256(str(len(df)).encode())
    digest.update(np.packbits(df[target_column].isna().to_numpy()).tobytes())
    if time_column:
        if time_column not in df.columns:
            raise ValueError(f"The time column {time_column!r} is not a column of this version")
        kind, values = time_axis(df[time_column])
        digest.update(kind.encode())
        digest.update(values.view(np.int64).tobytes() if kind == "time" else np.nan_to_num(values, nan=np.inf).tobytes())
    return digest.hexdigest()[:32]


def _bound(value: Any, kind: str, what: str) -> Any:
    if value is None:
        return None
    if kind == "index" or kind == "number":
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ValueError(f"The {what} of a range must be a number, got {value!r}") from None
    try:
        stamp = pd.Timestamp(str(value).strip())
    except (TypeError, ValueError):
        raise ValueError(f"The {what} of a range must be a time, got {value!r}") from None
    if stamp.tzinfo is not None:
        stamp = stamp.tz_localize(None)
    return np.datetime64(stamp.to_datetime64(), "ns")


def range_mask(df: pd.DataFrame, split_range: SplitRange, time_column: Optional[str]) -> np.ndarray:
    """The rows a range covers."""
    if split_range.by == "index":
        values = np.arange(len(df), dtype=float)
        kind = "index"
    else:
        if not time_column:
            raise ValueError("A range by time needs time_column")
        if time_column not in df.columns:
            raise ValueError(f"The time column {time_column!r} is not a column of this version")
        kind, values = time_axis(df[time_column])
    start = _bound(split_range.start, kind, "start")
    end = _bound(split_range.end, kind, "end")
    if start is not None and end is not None and end < start:
        start, end = end, start
    if kind == "time":
        mask = ~np.isnat(values)
    else:
        mask = ~np.isnan(values)
    if start is not None:
        mask &= values >= start
    if end is not None:
        mask &= values <= end
    return mask


def custom_labels(
    df: pd.DataFrame,
    eligible: np.ndarray,
    ranges: list[SplitRange],
    rest: Optional[str],
    time_column: Optional[str],
) -> np.ndarray:
    """Row labels from ranges painted in order (a later range wins), then ``rest`` for the rows left."""
    labels = np.full(len(df), -1, dtype=np.int8)
    for split_range in ranges:
        labels[range_mask(df, split_range, time_column)] = SET_CODES.get(split_range.set, UNUSED)
    labels[labels < 0] = SET_CODES[rest] if rest in SET_CODES else UNUSED
    labels[~eligible] = UNUSED
    return labels


def label_counts(labels: np.ndarray) -> dict[str, int]:
    return {
        **{name: int((labels == code).sum()) for name, code in SET_CODES.items()},
        "unused": int((labels == UNUSED).sum()),
    }


def check_labels(labels: np.ndarray) -> None:
    counts = label_counts(labels)
    if counts["train"] < 5:
        raise ValueError(f"The training set has {counts['train']} row(s); it needs at least 5")
    if counts["test"] < 2:
        raise ValueError(f"The test set has {counts['test']} row(s); it needs at least 2")
    if 0 < counts["validation"] < 2:
        raise ValueError("The validation set has 1 row; give it at least 2 or none")


def subset_summary(labels: np.ndarray, y: pd.Series) -> list[dict[str, Any]]:
    """Rows, share, target statistics and first / last row of each set."""
    used = int((labels != UNUSED).sum()) or 1
    numeric = pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y)
    out = []
    for name in SET_ORDER:
        rows = np.flatnonzero(labels == SET_CODES[name])
        if name == "validation" and not len(rows):
            continue
        part = y.iloc[rows] if numeric else pd.Series(dtype=float)
        out.append({
            "set": name,
            "rows": int(len(rows)),
            "share": round(len(rows) / used, 4),
            "target_mean": _finite(part.mean()),
            "target_std": _finite(part.std()),
            "target_min": _finite(part.min()),
            "target_max": _finite(part.max()),
            "first_row": int(rows.min()) if len(rows) else None,
            "last_row": int(rows.max()) if len(rows) else None,
        })
    return out


def _finite(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _write_labels(path: Any, labels: np.ndarray) -> None:
    with open(path, "wb") as handle:
        np.save(handle, labels.astype(np.int8), allow_pickle=False)


def load_labels(row: DataSplit) -> np.ndarray:
    return np.load(get_storage().fetch(row.artifact_key), allow_pickle=False)


def public(row: DataSplit) -> dict[str, Any]:
    """What tools and the UI get about a split."""
    meta = dict(row.meta or {})
    meta.pop("row_key", None)
    return {"split_id": str(row.id), **meta}


async def save_split(
    db: AsyncSession,
    *,
    version: DatasetVersion,
    labels: np.ndarray,
    meta: dict[str, Any],
) -> DataSplit:
    dataset = await db.get(Dataset, version.dataset_id)
    if dataset is None:
        raise ValueError(f"Dataset not found: {version.dataset_id}")
    split_id = uuid.uuid4()
    key = split_key(dataset.owner_id, dataset.id, split_id)
    get_storage().write(key, lambda path: _write_labels(path, labels))
    row = DataSplit(
        id=split_id,
        dataset_id=dataset.id,
        dataset_version_id=version.id,
        meta={**meta, "dataset_version_id": str(version.id), "created_at": datetime.now(timezone.utc).isoformat()},
        artifact_key=key,
    )
    db.add(row)
    await db.flush()
    await _prune(db, dataset.id)
    await db.commit()
    return row


async def _prune(db: AsyncSession, dataset_id: uuid.UUID) -> None:
    rows = (
        await db.execute(
            select(DataSplit).where(DataSplit.dataset_id == dataset_id).order_by(DataSplit.created_at.desc())
        )
    ).scalars().all()
    stale = rows[MAX_SPLITS_PER_DATASET:]
    if not stale:
        return
    await db.execute(delete(DataSplit).where(DataSplit.id.in_([r.id for r in stale])))
    get_storage().delete([r.artifact_key for r in stale])


async def dataset_splits(db: AsyncSession, dataset_id: uuid.UUID) -> list[DataSplit]:
    """Newest first."""
    rows = await db.execute(
        select(DataSplit)
        .where(DataSplit.dataset_id == dataset_id)
        .order_by(DataSplit.created_at.desc())
        .limit(_CANDIDATES)
    )
    return list(rows.scalars().all())


def fits(row: DataSplit, df: pd.DataFrame) -> Optional[str]:
    """Why the split does not fit the rows of ``df``, or None when it does."""
    meta = row.meta or {}
    target = meta.get("target_column")
    time_column = meta.get("time_column")
    if target not in df.columns:
        return f"its target {target!r} is not a column any more"
    if time_column and time_column not in df.columns:
        return f"its time column {time_column!r} is not a column any more"
    try:
        key = row_key(df, target, time_column)
    except ValueError as exc:
        return str(exc)
    if key != meta.get("row_key"):
        return "the rows changed since it was made (rows were dropped, added or reordered, or the target was filled in)"
    return None


async def current_split(
    db: AsyncSession,
    version: DatasetVersion,
    df: pd.DataFrame,
    target_column: Optional[str] = None,
) -> tuple[Optional[DataSplit], Optional[DataSplit], Optional[str]]:
    """(newest split that fits ``df``, newest split of the dataset when that one does not fit, why).

    With ``target_column`` only splits made for that target count.
    """
    rows = await dataset_splits(db, version.dataset_id)
    if target_column:
        rows = [r for r in rows if (r.meta or {}).get("target_column") == target_column]
    for row in rows:
        if fits(row, df) is None:
            return row, None, None
    if rows:
        return None, rows[0], fits(rows[0], df)
    return None, None, None


async def resolve_split(
    db: AsyncSession,
    version: DatasetVersion,
    df: pd.DataFrame,
    split_id: str,
    target_column: str,
) -> tuple[DataSplit, np.ndarray]:
    """The split ``split_id`` ('latest': the newest that fits) and its labels, checked against ``df``."""
    if split_id.strip().lower() == "latest":
        row, newest, why = await current_split(db, version, df, target_column)
        if row is None:
            if newest is not None:
                raise ValueError(f"The newest split for {target_column!r} no longer fits this data: {why}. Run split_data again")
            raise ValueError(f"There is no split for the target {target_column!r} yet; run split_data first")
    else:
        try:
            row = await db.get(DataSplit, uuid.UUID(split_id.strip()))
        except ValueError:
            row = None
        if row is None or row.dataset_id != version.dataset_id:
            raise ValueError(f"Split {split_id!r} not found for this dataset; run split_data, or pass split_id='latest'")
        meta = row.meta or {}
        if meta.get("target_column") != target_column:
            raise ValueError(
                f"Split {split_id[:8]} was made for the target {meta.get('target_column')!r}, not {target_column!r}; "
                "split again for this target"
            )
        why = fits(row, df)
        if why:
            raise ValueError(f"Split {split_id[:8]} no longer fits this data: {why}. Run split_data again")
    labels = load_labels(row)
    if len(labels) != len(df):
        raise ValueError("The stored split does not match the rows of this version; run split_data again")
    return row, labels


__all__ = [
    "MAX_SPLITS_PER_DATASET",
    "SET_CODES",
    "SET_NAMES",
    "SET_ORDER",
    "SplitRange",
    "UNUSED",
    "check_labels",
    "current_split",
    "custom_labels",
    "dataset_splits",
    "fits",
    "label_counts",
    "load_labels",
    "public",
    "range_mask",
    "resolve_split",
    "row_key",
    "save_split",
    "subset_summary",
    "time_axis",
]
