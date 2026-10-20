"""Row/column manipulation and feature tools that create new dataset versions.

Sorting, reshaping, combining datasets, where/mask, value mapping, function
application, time resampling and model-based feature selection. Each
mutating tool returns ``(new_version_id, summary)`` like the tools in
``eda_tools``; ``list_datasets`` returns a table.
"""

from __future__ import annotations

import math
import re
import uuid
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import AsyncSessionLocal
from backend.mcp_server.analysis_tools import AggFunc
from backend.mcp_server.eda_tools import (
    DropRowsArgs,
    RowOperator,
    _VALUELESS_OPERATORS,
    _assert_columns_exist,
    _assert_numeric_columns,
    _clean_names,
    _clean_optional_names,
    _condition_mask,
    _free_name,
    _get_version,
    _is_plain_numeric,
    _load_dataframe,
    _openai_tool_schema,
    _persist_new_version,
    _resolve_storage_path,
    _uniform_rows,
    _validate_uuid_str,
)
from backend.models import Dataset, DatasetVersion

_MAX_RESULT_ROWS = 1_000_000
_MAX_RESULT_COLUMNS = 2000
_MAX_PIVOT_VALUES = 200


def _blank(value: Any) -> Any:
    return value.strip() or None if isinstance(value, str) else value


async def _run_with(session: Optional[AsyncSession], run: Any) -> Any:
    if session is not None:
        return await run(session)
    async with AsyncSessionLocal() as db:
        return await run(db)


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------


class NaPosition(str, Enum):
    first = "first"
    last = "last"


class SortDataArgs(BaseModel):
    dataset_version_id: str = Field(..., min_length=36, max_length=36)
    by: list[str] = Field(..., min_length=1, description="Columns to sort by, most important first (sort_values).")
    ascending: bool = Field(True, description="Sort from smallest to largest (False: largest first).")
    descending_columns: Optional[list[str]] = Field(
        None,
        description="Columns of `by` sorted in the opposite direction to `ascending`, e.g. date ascending but score descending.",
    )
    na_position: NaPosition = Field(NaPosition.last, description="Put missing values first or last.")

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        return _validate_uuid_str(value)

    @field_validator("by")
    @classmethod
    def _names(cls, value: list[str]) -> list[str]:
        return _clean_names(value)

    @field_validator("descending_columns")
    @classmethod
    def _optional(cls, value: Optional[list[str]]) -> Optional[list[str]]:
        return _clean_optional_names(value)

    @model_validator(mode="after")
    def _subset(self) -> "SortDataArgs":
        extra = [c for c in (self.descending_columns or []) if c not in self.by]
        if extra:
            raise ValueError(f"descending_columns must be columns of by: {extra}")
        return self


async def sort_data(
    dataset_version_id: str,
    by: list[str],
    ascending: bool = True,
    descending_columns: Optional[list[str]] = None,
    na_position: str = "last",
    *,
    session: Optional[AsyncSession] = None,
) -> tuple[str, str]:
    """Reorder the rows by one or more columns (sort_values); the row index is the row order."""
    args = SortDataArgs(
        dataset_version_id=dataset_version_id, by=by, ascending=ascending,
        descending_columns=descending_columns, na_position=na_position,  # type: ignore[arg-type]
    )

    async def _run(db: AsyncSession) -> tuple[str, str]:
        parent = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(parent)
        _assert_columns_exist(df, args.by)
        flipped = set(args.descending_columns or [])
        directions = [args.ascending != (c in flipped) for c in args.by]
        ordered = df.sort_values(args.by, ascending=directions, na_position=args.na_position.value, kind="stable")
        described = ", ".join(f"{c} {'ascending' if up else 'descending'}" for c, up in zip(args.by, directions))
        if ordered.index.equals(df.index):
            return str(parent.id), f"Rows are already sorted by {described}. No changes made."
        return await _persist_new_version(
            db,
            parent=parent,
            df=ordered.reset_index(drop=True),
            action_performed="sort_data",
            parameters_used={"by": args.by, "ascending": directions, "na_position": args.na_position.value},
            summary_message=f"Sorted {len(df)} rows by {described} (missing values {args.na_position.value}).",
        )

    return await _run_with(session, _run)


# ---------------------------------------------------------------------------
# Reshaping
# ---------------------------------------------------------------------------


class ReshapeMode(str, Enum):
    melt = "melt"
    pivot = "pivot"
    stack = "stack"
    unstack = "unstack"


class ReshapeDataArgs(BaseModel):
    dataset_version_id: str = Field(..., min_length=36, max_length=36)
    mode: ReshapeMode = Field(
        ...,
        description=(
            "melt: wide to long — value_columns become (var_name, value_name) rows next to the "
            "id_columns (pd.melt); stack: like melt but keeps the original row number in a 'row' "
            "column, in row order (DataFrame.stack); pivot: long to wide — one row per id_columns "
            "combination and one column per value of pivot_column, filled with agg_func of the "
            "value column (pivot_table); unstack: long to wide without aggregation, each (id, "
            "pivot value) pair must occur once (DataFrame.pivot / unstack)."
        ),
    )
    id_columns: Optional[list[str]] = Field(
        None,
        description="melt/stack: columns kept as identifiers; pivot/unstack: the columns forming the rows (unstack defaults to 'row').",
    )
    value_columns: Optional[list[str]] = Field(
        None,
        description="melt/stack: columns turned into rows (omit for every non-id column); pivot/unstack: the column filling the cells (one).",
    )
    pivot_column: Optional[str] = Field(
        None,
        description="pivot/unstack: the column whose values become new columns (unstack defaults to var_name).",
    )
    var_name: str = Field("variable", description="melt/stack: name of the column holding the former column names.", min_length=1)
    value_name: str = Field("value", description="melt/stack: name of the column holding the values (unstack reads it by default).", min_length=1)
    agg_func: AggFunc = Field(AggFunc.mean, description="pivot: how repeated (id, pivot value) pairs are combined.")

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        return _validate_uuid_str(value)

    @field_validator("id_columns", "value_columns")
    @classmethod
    def _optional(cls, value: Optional[list[str]]) -> Optional[list[str]]:
        return _clean_optional_names(value)

    @field_validator("pivot_column", mode="before")
    @classmethod
    def _blank_pivot(cls, value: Any) -> Any:
        return _blank(value)

    @field_validator("var_name", "value_name")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("var_name and value_name cannot be blank")
        return value

    @model_validator(mode="after")
    def _mode_needs(self) -> "ReshapeDataArgs":
        if self.mode in (ReshapeMode.melt, ReshapeMode.stack):
            if self.var_name == self.value_name:
                raise ValueError("var_name and value_name must differ")
            clash = {self.var_name, self.value_name} & set(self.id_columns or [])
            if clash:
                raise ValueError(f"var_name / value_name clash with id_columns: {sorted(clash)}")
        if self.mode == ReshapeMode.pivot:
            if not self.id_columns:
                raise ValueError("pivot needs id_columns (the row keys)")
            if not self.pivot_column:
                raise ValueError("pivot needs pivot_column")
        if self.mode in (ReshapeMode.pivot, ReshapeMode.unstack) and self.value_columns and len(self.value_columns) > 1:
            raise ValueError(f"{self.mode.value} fills the cells from one value column")
        return self


def _wide(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    table.columns = [str(c) for c in table.columns]
    table.columns.name = None
    return table.reset_index()


def _reshape(df: pd.DataFrame, args: ReshapeDataArgs) -> tuple[pd.DataFrame, str]:
    mode = args.mode
    if mode in (ReshapeMode.melt, ReshapeMode.stack):
        ids = list(args.id_columns or [])
        _assert_columns_exist(df, ids)
        values = args.value_columns or [c for c in df.columns if c not in ids]
        _assert_columns_exist(df, values)
        if not values:
            raise ValueError(f"{mode.value} has no columns left to turn into rows")
        overlap = set(values) & set(ids)
        if overlap:
            raise ValueError(f"value_columns cannot also be id_columns: {sorted(overlap)}")
        if len(df) * len(values) > _MAX_RESULT_ROWS:
            raise ValueError(f"{mode.value} would create {len(df) * len(values)} rows (at most {_MAX_RESULT_ROWS})")
        if mode == ReshapeMode.melt:
            out = pd.melt(df, id_vars=ids, value_vars=values, var_name=args.var_name, value_name=args.value_name)
            return out, f"Melted {len(values)} column(s) into '{args.var_name}' / '{args.value_name}'"
        row = _free_name(df, "row")
        if row in (args.var_name, args.value_name):
            raise ValueError(f"var_name and value_name cannot be {row!r} in stack mode")
        framed = df.assign(**{row: np.arange(len(df))})
        out = pd.melt(framed, id_vars=[row, *ids], value_vars=values, var_name=args.var_name, value_name=args.value_name)
        out = out.sort_values(row, kind="stable").reset_index(drop=True)
        return out, f"Stacked {len(values)} column(s) into '{args.var_name}' / '{args.value_name}' with the row number in '{row}'"

    if mode == ReshapeMode.pivot:
        ids, pivot = list(args.id_columns or []), args.pivot_column
        _assert_columns_exist(df, [*ids, pivot])
        value = (args.value_columns or [None])[0]
        distinct = int(df[pivot].nunique(dropna=True))
        if distinct > _MAX_PIVOT_VALUES:
            raise ValueError(f"pivot_column {pivot!r} has {distinct} distinct values; at most {_MAX_PIVOT_VALUES} new columns")
        if value is None:
            table = pd.crosstab([df[c] for c in ids], df[pivot])
            described = "row counts"
        else:
            _assert_columns_exist(df, [value])
            if args.agg_func.value in ("sum", "mean", "median", "std", "var"):
                _assert_numeric_columns(df, [value])
            table = pd.pivot_table(df, index=ids, columns=pivot, values=value, aggfunc=args.agg_func.value, observed=True)
            described = f"{args.agg_func.value} of '{value}'"
        out = _wide(table)
        return out, f"Pivoted '{pivot}' into {out.shape[1] - len(ids)} column(s) of {described}, one row per {', '.join(ids)}"

    ids = list(args.id_columns or (["row"] if "row" in df.columns else []))
    if not ids:
        raise ValueError("unstack needs id_columns (or a 'row' column from stack)")
    pivot = args.pivot_column or args.var_name
    value = (args.value_columns or [args.value_name])[0]
    _assert_columns_exist(df, [*ids, pivot, value])
    duplicated = df.duplicated(subset=[*ids, pivot])
    if duplicated.any():
        raise ValueError(
            f"{int(duplicated.sum())} (id, {pivot}) pair(s) occur more than once; use pivot mode with an agg_func instead"
        )
    if int(df[pivot].nunique(dropna=True)) > _MAX_PIVOT_VALUES:
        raise ValueError(f"'{pivot}' has more than {_MAX_PIVOT_VALUES} distinct values")
    out = _wide(df.pivot(index=ids, columns=pivot, values=value))
    return out, f"Unstacked '{pivot}' into {out.shape[1] - len(ids)} column(s) of '{value}', one row per {', '.join(ids)}"


async def reshape_data(
    dataset_version_id: str,
    mode: str,
    id_columns: Optional[list[str]] = None,
    value_columns: Optional[list[str]] = None,
    pivot_column: Optional[str] = None,
    var_name: str = "variable",
    value_name: str = "value",
    agg_func: str = "mean",
    *,
    session: Optional[AsyncSession] = None,
) -> tuple[str, str]:
    """Reshape between wide and long layouts (melt, stack, pivot, unstack)."""
    args = ReshapeDataArgs(
        dataset_version_id=dataset_version_id, mode=mode, id_columns=id_columns,  # type: ignore[arg-type]
        value_columns=value_columns, pivot_column=pivot_column, var_name=var_name,
        value_name=value_name, agg_func=agg_func,  # type: ignore[arg-type]
    )

    async def _run(db: AsyncSession) -> tuple[str, str]:
        parent = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(parent)
        out, described = _reshape(df, args)
        if out.shape[1] > _MAX_RESULT_COLUMNS:
            raise ValueError(f"The result would have {out.shape[1]} columns (at most {_MAX_RESULT_COLUMNS})")
        return await _persist_new_version(
            db,
            parent=parent,
            df=out,
            action_performed="reshape_data",
            parameters_used=args.model_dump(mode="json", exclude={"dataset_version_id"}, exclude_none=True),
            summary_message=f"{described}: {df.shape[0]}×{df.shape[1]} → {out.shape[0]}×{out.shape[1]}.",
        )

    return await _run_with(session, _run)


# ---------------------------------------------------------------------------
# Combining datasets
# ---------------------------------------------------------------------------


class CombineMode(str, Enum):
    merge = "merge"
    join = "join"
    concat = "concat"


class JoinType(str, Enum):
    inner = "inner"
    left = "left"
    right = "right"
    outer = "outer"


class CombineDatasetsArgs(BaseModel):
    dataset_version_id: str = Field(..., min_length=36, max_length=36)
    other_dataset: str = Field(
        ...,
        description="The second dataset: its file name as uploaded (see list_datasets) or its dataset / version id.",
        min_length=1,
        max_length=512,
    )
    mode: CombineMode = Field(
        CombineMode.merge,
        description=(
            "merge: SQL-style join on the key columns `on` (pd.merge); join: put the other "
            "dataset's columns next to these, matching rows by position (DataFrame.join); concat: "
            "append the other dataset's rows below these (pd.concat)."
        ),
    )
    on: Optional[list[str]] = Field(None, description="For merge: key columns present in both datasets.")
    join_type: JoinType = Field(
        JoinType.inner,
        description=(
            "merge and join: inner (only matches), left (all rows of this dataset), right, or outer "
            "(all rows of both). concat: inner keeps only the shared columns, outer keeps all."
        ),
    )
    suffix: str = Field("_other", description="Added to the other dataset's column names that clash with this one's.", min_length=1, max_length=30)
    add_source_column: bool = Field(False, description="For concat: add a 'source' column naming the dataset each row came from.")

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        return _validate_uuid_str(value)

    @field_validator("other_dataset", "suffix")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("other_dataset and suffix cannot be blank")
        return value

    @field_validator("on")
    @classmethod
    def _optional(cls, value: Optional[list[str]]) -> Optional[list[str]]:
        return _clean_optional_names(value)

    @model_validator(mode="after")
    def _mode_needs(self) -> "CombineDatasetsArgs":
        if self.mode == CombineMode.merge and not self.on:
            raise ValueError("merge needs the key columns in `on`")
        if self.mode == CombineMode.concat and self.join_type not in (JoinType.inner, JoinType.outer):
            raise ValueError("concat takes join_type inner (shared columns) or outer (all columns)")
        return self


async def _latest_version(db: AsyncSession, dataset_id: uuid.UUID) -> Optional[DatasetVersion]:
    """The newest version of a dataset (a version no other version descends from, if times tie)."""
    versions = (
        await db.execute(select(DatasetVersion).where(DatasetVersion.dataset_id == dataset_id))
    ).scalars().all()
    if not versions:
        return None
    parents = {v.parent_version_id for v in versions}
    return max(versions, key=lambda v: (v.created_at, v.id not in parents))


async def _owner_of(db: AsyncSession, version: DatasetVersion) -> str:
    dataset = await db.get(Dataset, version.dataset_id)
    if dataset is None:
        raise ValueError(f"Dataset not found: {version.dataset_id}")
    return dataset.owner_id


async def owner_datasets(db: AsyncSession, owner_id: str) -> list[tuple[Dataset, DatasetVersion]]:
    """Every dataset of the owner with its latest version, newest upload first."""
    datasets = (
        await db.execute(select(Dataset).where(Dataset.owner_id == owner_id).order_by(Dataset.created_at.desc()))
    ).scalars().all()
    out = []
    for dataset in datasets:
        latest = await _latest_version(db, dataset.id)
        if latest is not None:
            out.append((dataset, latest))
    return out


async def resolve_other_dataset(db: AsyncSession, active: DatasetVersion, reference: str) -> tuple[Dataset, DatasetVersion]:
    """The owner's dataset version named by file name, dataset id or version id.

    Datasets of other owners are reported exactly like missing ones.
    """
    owner = await _owner_of(db, active)
    not_found = ValueError(
        f"No uploaded dataset of yours matches {reference!r}. Upload it first; list_datasets shows the uploaded files."
    )
    try:
        ref_id = uuid.UUID(reference)
    except ValueError:
        ref_id = None
    if ref_id is not None:
        version = await db.get(DatasetVersion, ref_id)
        if version is not None:
            dataset = await db.get(Dataset, version.dataset_id)
            if dataset is not None and dataset.owner_id == owner:
                return dataset, version
            raise not_found
        dataset = await db.get(Dataset, ref_id)
        if dataset is not None and dataset.owner_id == owner:
            latest = await _latest_version(db, dataset.id)
            if latest is not None:
                return dataset, latest
        raise not_found
    wanted = reference.casefold()
    for dataset, latest in await owner_datasets(db, owner):
        if dataset.file_name.casefold() == wanted or dataset.file_name.rsplit(".", 1)[0].casefold() == wanted:
            return dataset, latest
    raise not_found


def _merge_size(left: pd.DataFrame, right: pd.DataFrame, on: list[str], how: JoinType) -> int:
    """Exact row count of pd.merge (missing keys match each other, as in pandas)."""
    lc = left.groupby(on, dropna=False, observed=True).size()
    rc = right.groupby(on, dropna=False, observed=True).size()
    both = lc.to_frame("l").join(rc.to_frame("r"), how="outer").fillna(0)
    matched = int((both["l"] * both["r"]).sum())
    left_only = int(both.loc[both["r"] == 0, "l"].sum())
    right_only = int(both.loc[both["l"] == 0, "r"].sum())
    return {
        JoinType.inner: matched,
        JoinType.left: matched + left_only,
        JoinType.right: matched + right_only,
        JoinType.outer: matched + left_only + right_only,
    }[how]


def _combine(df: pd.DataFrame, other: pd.DataFrame, args: CombineDatasetsArgs, names: tuple[str, str]) -> tuple[pd.DataFrame, str]:
    if args.mode == CombineMode.merge:
        missing_here = [c for c in args.on if c not in df.columns]
        missing_there = [c for c in args.on if c not in other.columns]
        if missing_here or missing_there:
            raise ValueError(
                "Key columns must exist in both datasets"
                + (f"; missing here: {missing_here}" if missing_here else "")
                + (f"; missing in {names[1]}: {missing_there}" if missing_there else "")
            )
        size = _merge_size(df, other, args.on, args.join_type)
        if size > _MAX_RESULT_ROWS:
            raise ValueError(f"The merge would create {size} rows (at most {_MAX_RESULT_ROWS}); the keys repeat too often")
        try:
            out = pd.merge(df, other, on=args.on, how=args.join_type.value, suffixes=("", args.suffix))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Could not merge on {args.on}: {exc}") from exc
        return out, f"Merged with {names[1]} on {', '.join(args.on)} ({args.join_type.value} join)"
    if args.mode == CombineMode.join:
        clash = [c for c in other.columns if c in df.columns]
        renamed = other.rename(columns={c: f"{c}{args.suffix}" for c in clash})
        dup = [c for c in renamed.columns if c in df.columns]
        if dup:
            raise ValueError(f"Columns still clash after adding the suffix: {dup}; choose another suffix")
        out = df.reset_index(drop=True).join(renamed.reset_index(drop=True), how=args.join_type.value)
        return out.reset_index(drop=True), f"Joined {names[1]}'s columns by row position ({args.join_type.value})"
    frames = [df, other]
    if args.add_source_column:
        source = _free_name(pd.DataFrame(columns=[*df.columns, *other.columns]), "source")
        frames = [df.assign(**{source: names[0]}), other.assign(**{source: names[1]})]
    out = pd.concat(frames, join=args.join_type.value, ignore_index=True)
    shared = len(set(df.columns) & set(other.columns))
    return out, f"Appended {len(other)} row(s) of {names[1]} ({args.join_type.value}: {shared} shared column(s))"


async def combine_datasets(
    dataset_version_id: str,
    other_dataset: str,
    mode: str = "merge",
    on: Optional[list[str]] = None,
    join_type: str = "inner",
    suffix: str = "_other",
    add_source_column: bool = False,
    *,
    session: Optional[AsyncSession] = None,
) -> tuple[str, str]:
    """Merge, join or concatenate this dataset with another dataset of the same user."""
    args = CombineDatasetsArgs(
        dataset_version_id=dataset_version_id, other_dataset=other_dataset, mode=mode,  # type: ignore[arg-type]
        on=on, join_type=join_type, suffix=suffix, add_source_column=add_source_column,  # type: ignore[arg-type]
    )

    async def _run(db: AsyncSession) -> tuple[str, str]:
        parent = await _get_version(db, args.dataset_version_id)
        this_dataset = await db.get(Dataset, parent.dataset_id)
        other_dataset_row, other_version = await resolve_other_dataset(db, parent, args.other_dataset)
        df = _load_dataframe(parent)
        other = _load_dataframe(other_version)
        names = (this_dataset.file_name if this_dataset else "this dataset", other_dataset_row.file_name)
        out, described = _combine(df, other, args, names)
        if out.shape[1] > _MAX_RESULT_COLUMNS:
            raise ValueError(f"The result would have {out.shape[1]} columns (at most {_MAX_RESULT_COLUMNS})")
        if len(out) > _MAX_RESULT_ROWS:
            raise ValueError(f"The result would have {len(out)} rows (at most {_MAX_RESULT_ROWS})")
        if out.empty:
            raise ValueError(f"{described} leaves no rows; check the keys or use another join_type")
        return await _persist_new_version(
            db,
            parent=parent,
            df=out,
            action_performed="combine_datasets",
            parameters_used={
                **args.model_dump(mode="json", exclude={"dataset_version_id", "other_dataset"}, exclude_none=True),
                "other_dataset": other_dataset_row.file_name,
                "other_dataset_version_id": str(other_version.id),
            },
            summary_message=f"{described}: {df.shape[0]}×{df.shape[1]} → {out.shape[0]}×{out.shape[1]}.",
        )

    return await _run_with(session, _run)


class ListDatasetsArgs(BaseModel):
    dataset_version_id: str = Field(..., min_length=36, max_length=36)

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        return _validate_uuid_str(value)


async def list_datasets(dataset_version_id: str, *, session: Optional[AsyncSession] = None) -> dict[str, Any]:
    """The user's uploaded datasets (file name, size, latest version) for combine_datasets."""
    args = ListDatasetsArgs(dataset_version_id=dataset_version_id)

    async def _run(db: AsyncSession) -> dict[str, Any]:
        active = await _get_version(db, args.dataset_version_id)
        rows = []
        for dataset, latest in await owner_datasets(db, await _owner_of(db, active)):
            try:
                parquet = pq.ParquetFile(_resolve_storage_path(latest.file_path))
                columns = [f for f in parquet.schema_arrow if not f.name.startswith("__index_level")]
                shape = {"rows": parquet.metadata.num_rows, "columns": len(columns)}
            except (FileNotFoundError, OSError):
                shape = {"rows": None, "columns": None}
            rows.append({
                "file_name": dataset.file_name,
                **shape,
                "current": dataset.id == active.dataset_id,
                "latest_version_id": str(latest.id),
                "uploaded": dataset.created_at.isoformat() if dataset.created_at else None,
            })
        return {"dataset_version_id": args.dataset_version_id, "title": f"Your datasets ({len(rows)})", "rows": _uniform_rows(rows)}

    return await _run_with(session, _run)


# ---------------------------------------------------------------------------
# where / mask
# ---------------------------------------------------------------------------


class MaskAction(str, Enum):
    mask = "mask"
    where = "where"


class MaskValuesArgs(BaseModel):
    dataset_version_id: str = Field(..., min_length=36, max_length=36)
    columns: list[str] = Field(..., min_length=1, description="Columns whose values are replaced.")
    action: MaskAction = Field(
        MaskAction.mask,
        description="mask: replace the values where the condition holds (Series.mask); where: keep them and replace the rest (Series.where).",
    )
    condition_column: Optional[str] = Field(
        None,
        description="Column the condition tests. Omit to test each of `columns` itself.",
    )
    operator: RowOperator = Field(
        RowOperator.greater_than,
        description="equals, not_equals, greater_than, greater_or_equal, less_than, less_or_equal, contains, is_missing, not_missing.",
    )
    value: Optional[str] = Field(None, description="Value the condition compares with (a number for numeric columns).")
    replacement: Optional[str] = Field(
        None,
        description="What replaced values become (a number for numeric columns). Omit to make them missing.",
    )

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        return _validate_uuid_str(value)

    @field_validator("columns")
    @classmethod
    def _names(cls, value: list[str]) -> list[str]:
        return _clean_names(value)

    @field_validator("condition_column", mode="before")
    @classmethod
    def _blank_condition(cls, value: Any) -> Any:
        return _blank(value)

    @field_validator("value", "replacement", mode="before")
    @classmethod
    def _as_text(cls, value: Any) -> Any:
        return None if value is None else str(value)

    @model_validator(mode="after")
    def _needs_value(self) -> "MaskValuesArgs":
        if self.operator not in _VALUELESS_OPERATORS and (self.value is None or self.value == ""):
            raise ValueError(f"The {self.operator.value} operator needs a value")
        return self


def _typed_replacement(series: pd.Series, text: Optional[str]) -> Any:
    if text is None:
        return np.nan
    if pd.api.types.is_bool_dtype(series):
        lowered = text.strip().lower()
        if lowered not in {"true", "false", "1", "0", "yes", "no"}:
            raise ValueError(f"'{series.name}' is true/false, so replacement must be true or false")
        return lowered in {"true", "1", "yes"}
    if pd.api.types.is_numeric_dtype(series):
        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(f"'{series.name}' is numeric, so replacement must be a number (got {text!r})") from exc
    if pd.api.types.is_datetime64_any_dtype(series):
        try:
            return pd.Timestamp(text)
        except ValueError as exc:
            raise ValueError(f"'{series.name}' holds dates, so replacement must be a date (got {text!r})") from exc
    return text


async def mask_values(
    dataset_version_id: str,
    columns: list[str],
    action: str = "mask",
    condition_column: Optional[str] = None,
    operator: str = "greater_than",
    value: Optional[str] = None,
    replacement: Optional[str] = None,
    *,
    session: Optional[AsyncSession] = None,
) -> tuple[str, str]:
    """Replace values where a condition holds (mask) or does not hold (where)."""
    args = MaskValuesArgs(
        dataset_version_id=dataset_version_id, columns=columns, action=action,  # type: ignore[arg-type]
        condition_column=condition_column, operator=operator, value=value,  # type: ignore[arg-type]
        replacement=replacement,
    )

    async def _run(db: AsyncSession) -> tuple[str, str]:
        parent = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(parent)
        _assert_columns_exist(df, [*args.columns, *([args.condition_column] if args.condition_column else [])])
        result = df.copy()
        changed = 0
        for col in args.columns:
            tested = args.condition_column or col
            condition = _condition_mask(df, DropRowsArgs(
                dataset_version_id=args.dataset_version_id, mode="condition", column=tested,
                operator=args.operator, value=args.value,
            ))
            replace = condition if args.action == MaskAction.mask else ~condition
            original = df[col]
            new_value = _typed_replacement(original, args.replacement)
            if pd.api.types.is_integer_dtype(original) and isinstance(new_value, float) and new_value.is_integer():
                new_value = int(new_value)
            if isinstance(original.dtype, pd.CategoricalDtype):
                updated = original.astype(object).mask(replace, new_value).astype("category")
            else:
                updated = original.mask(replace, new_value)
            differs = ~((original.astype(object) == updated.astype(object)) | (original.isna() & updated.isna()))
            changed += int(differs.sum())
            result[col] = updated
        rule = f"{args.condition_column or 'the value'} {args.operator.value}" + (
            f" {args.value}" if args.operator not in _VALUELESS_OPERATORS else ""
        )
        target = "missing" if args.replacement is None else repr(args.replacement)
        where = "where" if args.action == MaskAction.mask else "where not"
        if changed == 0:
            return str(parent.id), f"No values to replace ({where} {rule}). No changes made."
        return await _persist_new_version(
            db,
            parent=parent,
            df=result,
            action_performed="mask_values",
            parameters_used={**args.model_dump(mode="json", exclude={"dataset_version_id"}), "values_changed": changed},
            summary_message=f"Set {changed} value(s) in {', '.join(args.columns)} to {target} {where} {rule} ({args.action.value}).",
        )

    return await _run_with(session, _run)


# ---------------------------------------------------------------------------
# map
# ---------------------------------------------------------------------------


class UnmappedValues(str, Enum):
    keep = "keep"
    missing = "missing"


class MapValuesArgs(BaseModel):
    dataset_version_id: str = Field(..., min_length=36, max_length=36)
    columns: list[str] = Field(..., min_length=1, description="Columns whose values are mapped.")
    value_mapping: dict[str, str] = Field(
        ...,
        description="Old value → new value, e.g. {\"low\": \"1\", \"high\": \"3\"} (Series.map). New values that are all numbers give a numeric column.",
    )
    unmapped: UnmappedValues = Field(UnmappedValues.keep, description="Values not in value_mapping: keep them, or make them missing.")

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        return _validate_uuid_str(value)

    @field_validator("columns")
    @classmethod
    def _names(cls, value: list[str]) -> list[str]:
        return _clean_names(value)

    @field_validator("value_mapping", mode="before")
    @classmethod
    def _text_mapping(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        cleaned = {str(k).strip(): "" if v is None else str(v).strip() for k, v in value.items() if str(k).strip()}
        if not cleaned:
            raise ValueError("value_mapping needs at least one old → new pair")
        return cleaned


def _canonical(value: Any) -> Optional[str]:
    """A lookup key that makes 1, 1.0 and "1" (and True / "true") the same value."""
    if value is None or (isinstance(value, float) and math.isnan(value)) or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return str(int(number)) if number.is_integer() else repr(number)
    text = str(value).strip()
    if text.lower() in {"true", "false"}:
        return text.lower()
    try:
        number = float(text)
    except ValueError:
        return text
    if math.isfinite(number):
        return str(int(number)) if number.is_integer() else repr(number)
    return text


def _mapped_target(text: str) -> Any:
    if text == "":
        return np.nan
    try:
        number = float(text)
    except ValueError:
        return text
    return int(number) if number.is_integer() and re.fullmatch(r"[+-]?\d+", text) else number


async def map_values(
    dataset_version_id: str,
    columns: list[str],
    value_mapping: dict[str, str],
    unmapped: str = "keep",
    *,
    session: Optional[AsyncSession] = None,
) -> tuple[str, str]:
    """Replace values through a lookup table (Series.map / replace)."""
    args = MapValuesArgs(
        dataset_version_id=dataset_version_id, columns=columns, value_mapping=value_mapping,
        unmapped=unmapped,  # type: ignore[arg-type]
    )
    lookup = {_canonical(k): _mapped_target(v) for k, v in args.value_mapping.items()}

    async def _run(db: AsyncSession) -> tuple[str, str]:
        parent = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(parent)
        _assert_columns_exist(df, args.columns)
        result = df.copy()
        changed = 0
        unknown: dict[str, int] = {}
        for col in args.columns:
            original = df[col]
            keys = original.astype(object).map(_canonical)
            hit = keys.map(lambda k: k in lookup).astype(bool)
            mapped = keys.map(lambda k: lookup.get(k)).astype(object)
            fallback = original.astype(object) if args.unmapped == UnmappedValues.keep else None
            chosen = mapped.where(hit, fallback)
            values = pd.Series([None if pd.isna(v) else v for v in chosen], index=original.index)
            missed = int((~hit & keys.notna()).sum())
            if missed:
                unknown[col] = missed
            differs = ~((original.astype(object) == values.astype(object)) | (original.isna() & values.isna()))
            changed += int(differs.sum())
            result[col] = values
        if changed == 0:
            return str(parent.id), "No values matched value_mapping. No changes made."
        summary = f"Mapped {changed} value(s) in {', '.join(args.columns)} with {len(lookup)} rule(s)."
        if unknown:
            fate = "kept" if args.unmapped == UnmappedValues.keep else "made missing"
            summary += " Values without a rule were " + fate + ": " + ", ".join(f"{c} ({n})" for c, n in unknown.items()) + "."
        return await _persist_new_version(
            db,
            parent=parent,
            df=result,
            action_performed="map_values",
            parameters_used={**args.model_dump(mode="json", exclude={"dataset_version_id"}), "values_changed": changed},
            summary_message=summary,
        )

    return await _run_with(session, _run)


# ---------------------------------------------------------------------------
# apply / transform
# ---------------------------------------------------------------------------


class ApplyFunction(str, Enum):
    abs = "abs"
    round = "round"
    floor = "floor"
    ceil = "ceil"
    sign = "sign"
    negate = "negate"
    square = "square"
    sqrt = "sqrt"
    reciprocal = "reciprocal"
    exp = "exp"
    log = "log"
    log10 = "log10"
    clip_negative = "clip_negative"
    cumsum = "cumsum"
    cumprod = "cumprod"
    cummax = "cummax"
    cummin = "cummin"
    rank = "rank"
    pct_rank = "pct_rank"
    zscore = "zscore"
    minmax = "minmax"
    demean = "demean"
    group_mean = "group_mean"
    group_sum = "group_sum"
    group_count = "group_count"
    group_min = "group_min"
    group_max = "group_max"
    group_std = "group_std"
    group_median = "group_median"


_ELEMENTWISE = frozenset({
    ApplyFunction.abs, ApplyFunction.round, ApplyFunction.floor, ApplyFunction.ceil, ApplyFunction.sign,
    ApplyFunction.negate, ApplyFunction.square, ApplyFunction.sqrt, ApplyFunction.reciprocal,
    ApplyFunction.exp, ApplyFunction.log, ApplyFunction.log10, ApplyFunction.clip_negative,
})


class ApplyFunctionArgs(BaseModel):
    dataset_version_id: str = Field(..., min_length=36, max_length=36)
    columns: list[str] = Field(..., min_length=1, description="Numeric columns the function is applied to.")
    function: ApplyFunction = Field(
        ...,
        description=(
            "Element-wise (apply / map): abs, round, floor, ceil, sign, negate, square, sqrt, "
            "reciprocal, exp, log (natural), log10, clip_negative (negatives → 0). Over the column, "
            "or per group with group_by (transform): cumsum, cumprod, cummax, cummin, rank, "
            "pct_rank, zscore, minmax (0–1), demean, group_mean, group_sum, group_count, group_min, "
            "group_max, group_std, group_median (each row gets its group's statistic)."
        ),
    )
    group_by: Optional[list[str]] = Field(
        None,
        description="For the column-wide functions: compute them within each group of these columns (groupby().transform()).",
    )
    decimals: int = Field(2, description="For round: decimal places.", ge=0, le=12)
    in_place: bool = Field(False, description="Overwrite the columns instead of adding <col>_<function> columns.")

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        return _validate_uuid_str(value)

    @field_validator("columns")
    @classmethod
    def _names(cls, value: list[str]) -> list[str]:
        return _clean_names(value)

    @field_validator("group_by")
    @classmethod
    def _optional(cls, value: Optional[list[str]]) -> Optional[list[str]]:
        return _clean_optional_names(value)

    @model_validator(mode="after")
    def _grouping(self) -> "ApplyFunctionArgs":
        if self.group_by and self.function in _ELEMENTWISE:
            raise ValueError(f"{self.function.value} works value by value; group_by applies to the column-wide functions")
        overlap = set(self.group_by or []) & set(self.columns)
        if overlap:
            raise ValueError(f"group_by cannot include the transformed columns: {sorted(overlap)}")
        if self.function.value.startswith("group_") and not self.group_by:
            raise ValueError(f"{self.function.value} needs group_by")
        return self


def _elementwise(x: pd.Series, fn: ApplyFunction, decimals: int) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out = {
            ApplyFunction.abs: lambda: x.abs(),
            ApplyFunction.round: lambda: x.round(decimals),
            ApplyFunction.floor: lambda: np.floor(x),
            ApplyFunction.ceil: lambda: np.ceil(x),
            ApplyFunction.sign: lambda: np.sign(x),
            ApplyFunction.negate: lambda: -x,
            ApplyFunction.square: lambda: x ** 2,
            ApplyFunction.sqrt: lambda: np.sqrt(x.where(x >= 0)),
            ApplyFunction.reciprocal: lambda: 1.0 / x.replace(0, np.nan),
            ApplyFunction.exp: lambda: np.exp(x),
            ApplyFunction.log: lambda: np.log(x.where(x > 0)),
            ApplyFunction.log10: lambda: np.log10(x.where(x > 0)),
            ApplyFunction.clip_negative: lambda: x.clip(lower=0),
        }[fn]()
    return out.replace([np.inf, -np.inf], np.nan)


def _columnwise(x: pd.Series, fn: ApplyFunction) -> pd.Series:
    if fn in (ApplyFunction.cumsum, ApplyFunction.cumprod, ApplyFunction.cummax, ApplyFunction.cummin):
        return getattr(x, fn.value)()
    if fn == ApplyFunction.rank:
        return x.rank(method="average")
    if fn == ApplyFunction.pct_rank:
        return x.rank(method="average", pct=True)
    if fn == ApplyFunction.zscore:
        std = x.std()
        return (x - x.mean()) / std if std and np.isfinite(std) else x * np.nan
    if fn == ApplyFunction.minmax:
        span = x.max() - x.min()
        return (x - x.min()) / span if span and np.isfinite(span) else x * np.nan
    if fn == ApplyFunction.demean:
        return x - x.mean()
    stat = fn.value.removeprefix("group_")
    return pd.Series(getattr(x, stat)(), index=x.index, dtype=float)


async def apply_function(
    dataset_version_id: str,
    columns: list[str],
    function: str,
    group_by: Optional[list[str]] = None,
    decimals: int = 2,
    in_place: bool = False,
    *,
    session: Optional[AsyncSession] = None,
) -> tuple[str, str]:
    """Apply an element-wise, cumulative, ranking, scaling or group function to numeric columns."""
    args = ApplyFunctionArgs(
        dataset_version_id=dataset_version_id, columns=columns, function=function,  # type: ignore[arg-type]
        group_by=group_by, decimals=decimals, in_place=in_place,
    )
    fn = args.function

    async def _run(db: AsyncSession) -> tuple[str, str]:
        parent = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(parent)
        _assert_numeric_columns(df, args.columns)
        if args.group_by:
            _assert_columns_exist(df, args.group_by)
        result = df.copy()
        written: list[str] = []
        made_missing = 0
        for col in args.columns:
            x = df[col].astype(float)
            if fn in _ELEMENTWISE:
                out = _elementwise(x, fn, args.decimals)
            elif args.group_by:
                out = x.groupby([df[g] for g in args.group_by], dropna=False, observed=True).transform(
                    lambda part: _columnwise(part, fn)
                )
            else:
                out = _columnwise(x, fn)
            made_missing += int((x.notna() & out.isna()).sum())
            name = col if args.in_place else f"{col}_{fn.value}" + (f"_by_{'_'.join(args.group_by)}" if args.group_by else "")
            result[name] = out.to_numpy()
            written.append(name)
        scope = f" within groups of {', '.join(args.group_by)}" if args.group_by else ""
        summary = f"Applied {fn.value}{scope} to {', '.join(args.columns)}" + (
            " in place." if args.in_place else f"; added {', '.join(written)}."
        )
        if made_missing:
            summary += f" {made_missing} value(s) outside the function's domain became missing."
        return await _persist_new_version(
            db,
            parent=parent,
            df=result,
            action_performed="apply_function",
            parameters_used=args.model_dump(mode="json", exclude={"dataset_version_id"}, exclude_none=True),
            summary_message=summary,
        )

    return await _run_with(session, _run)


# ---------------------------------------------------------------------------
# Time resampling
# ---------------------------------------------------------------------------


class ResampleMode(str, Enum):
    downsample = "downsample"
    upsample = "upsample"


class ResampleAggregation(str, Enum):
    mean = "mean"
    median = "median"
    sum = "sum"
    min = "min"
    max = "max"
    first = "first"
    last = "last"
    std = "std"
    count = "count"


class UpsampleFill(str, Enum):
    interpolate = "interpolate"
    ffill = "ffill"
    bfill = "bfill"
    nearest = "nearest"
    none = "none"


_FREQUENCY = re.compile(r"^\s*(\d+(?:\.\d+)?)?\s*([A-Za-z]+)\s*$")
_FREQUENCY_UNITS = {
    **dict.fromkeys(("s", "sec", "secs", "second", "seconds"), "s"),
    **dict.fromkeys(("min", "mins", "minute", "minutes", "m", "t"), "min"),
    **dict.fromkeys(("h", "hr", "hrs", "hour", "hours"), "h"),
    **dict.fromkeys(("d", "day", "days"), "D"),
    **dict.fromkeys(("w", "wk", "week", "weeks"), "W"),
    **dict.fromkeys(("mo", "month", "months"), "MS"),
}
_FILL_DESCRIPTIONS = {
    UpsampleFill.interpolate: "linear interpolation in time",
    UpsampleFill.ffill: "carrying the last value forward",
    UpsampleFill.bfill: "carrying the next value backward",
    UpsampleFill.nearest: "the nearest measured value",
    UpsampleFill.none: "nothing (new time steps stay missing)",
}


def _offset(text: str) -> pd.DateOffset:
    """A pandas offset from '5min', '1h', '15 minutes', '1D', '1MS'…"""
    from pandas.tseries.frequencies import to_offset

    offset = None
    try:
        offset = to_offset(re.sub(r"\s+", "", text))
    except (ValueError, TypeError):
        match = _FREQUENCY.match(text)
        unit = _FREQUENCY_UNITS.get(match.group(2).lower()) if match else None
        if unit is not None:
            offset = to_offset(f"{match.group(1) or 1}{unit}")
    if offset is None:
        raise ValueError(
            f"frequency {text!r} is not understood; use a number and a unit such as 30s, 5min, 1h, 1D, 1W or 1MS"
        )
    if offset.n <= 0:
        raise ValueError("frequency must be a positive time step")
    return offset


def _step_text(step: pd.Timedelta) -> str:
    seconds = step.total_seconds()
    for unit, size in (("d", 86400), ("h", 3600), ("min", 60)):
        if seconds >= size:
            return f"{seconds / size:.3g} {unit}"
    return f"{seconds:.3g} s"


def _fixed_step(offset: pd.DateOffset) -> Optional[pd.Timedelta]:
    try:
        return pd.Timedelta(offset)
    except (ValueError, TypeError):
        return None


class ResampleTimeSeriesArgs(BaseModel):
    dataset_version_id: str = Field(..., min_length=36, max_length=36)
    time_column: str = Field(..., description="Column holding the timestamps (dates / times) of the rows.", min_length=1)
    frequency: str = Field(
        ...,
        description=(
            "The new time step: a number and a unit, e.g. 30s, 5min, 15min, 1h, 8h, 1D, 1W or 1MS "
            "(month start). Units: s, min, h, D (day), W (week), MS (month)."
        ),
        min_length=1,
        max_length=20,
    )
    mode: ResampleMode = Field(
        ResampleMode.downsample,
        description=(
            "downsample: a coarser step; the rows in each interval are combined with `aggregation` "
            "(e.g. 1-min historian data to hourly means). upsample: a finer step; new time steps are "
            "filled with `fill` (e.g. 4-hourly lab results onto a 1-min grid). A step equal to the "
            "data's spacing puts irregular timestamps on a regular grid."
        ),
    )
    aggregation: ResampleAggregation = Field(
        ResampleAggregation.mean,
        description=(
            "downsample: how the numeric values in each interval are combined: mean, median, sum, "
            "min, max, first, last, std or count. Text columns take the first value (last for "
            "'last', the number of values for 'count')."
        ),
    )
    fill: UpsampleFill = Field(
        UpsampleFill.interpolate,
        description=(
            "upsample: how new time steps are filled: interpolate (linear in time between measured "
            "values; text columns carry the last value), ffill (carry the last value forward, e.g. a "
            "lab result held until the next sample), bfill, nearest, or none (leave them missing)."
        ),
    )
    fill_limit: Optional[int] = Field(
        None,
        description="upsample: fill at most this many consecutive new time steps, so a long outage is not invented. Omit for no limit.",
        ge=1,
    )
    columns: Optional[list[str]] = Field(None, description="Columns to keep besides the time column. Omit to keep all.")
    drop_empty: bool = Field(
        True,
        description="downsample: drop intervals that contain no rows (otherwise they stay as rows of missing values).",
    )

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        return _validate_uuid_str(value)

    @field_validator("time_column")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("time_column cannot be blank")
        return value

    @field_validator("frequency")
    @classmethod
    def _valid_frequency(cls, value: str) -> str:
        _offset(value)
        return value.strip()

    @field_validator("columns")
    @classmethod
    def _optional(cls, value: Optional[list[str]]) -> Optional[list[str]]:
        return _clean_optional_names(value)


def _timestamps(series: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    if _is_plain_numeric(series):
        raise ValueError(f"'{name}' holds numbers, not dates or times; convert it to datetime first (convert_types)")
    return pd.to_datetime(series, errors="coerce", format="mixed")


def _downsample(frame: pd.DataFrame, offset: pd.DateOffset, args: ResampleTimeSeriesArgs, numeric: list[str], other: list[str]) -> tuple[pd.DataFrame, str]:
    bins = frame.resample(offset)
    agg = args.aggregation
    pieces = []
    if numeric:
        selected = bins[numeric]
        pieces.append(selected.sum(min_count=1) if agg == ResampleAggregation.sum else getattr(selected, agg.value)())
    other_agg = {ResampleAggregation.count: "count", ResampleAggregation.last: "last"}.get(agg, "first")
    if other:
        pieces.append(getattr(bins[other], other_agg)())
    out = pd.concat(pieces, axis=1)[[*numeric, *other]]
    sizes = bins.size().reindex(out.index).fillna(0)
    empty = int((sizes == 0).sum())
    described = f"{agg.value} of {len(numeric)} numeric column(s)"
    if other:
        described += f"; text columns take the {other_agg} {'number of values' if other_agg == 'count' else 'value'}"
    if args.drop_empty and empty:
        out = out.loc[sizes > 0]
        described += f"; {empty} empty interval(s) dropped"
    elif empty:
        described += f"; {empty} empty interval(s) kept as missing rows"
    return out, described


def _upsample(frame: pd.DataFrame, offset: pd.DateOffset, args: ResampleTimeSeriesArgs, numeric: list[str], other: list[str]) -> tuple[pd.DataFrame, str]:
    described = f"filled by {_FILL_DESCRIPTIONS[args.fill]}"
    if frame.index.has_duplicates:
        repeated = int(frame.index.duplicated().sum())
        grouped = frame.groupby(level=0)
        parts = ([grouped[numeric].mean()] if numeric else []) + ([grouped[other].first()] if other else [])
        frame = pd.concat(parts, axis=1)
        described += f"; {repeated} row(s) sharing a timestamp were averaged first"
    step = _fixed_step(offset)
    if step is not None and (frame.index[-1] - frame.index[0]) / step + 1 > _MAX_RESULT_ROWS:
        raise ValueError(f"Upsampling to {args.frequency} would create more than {_MAX_RESULT_ROWS:,} rows; choose a coarser step")
    grid = frame.resample(offset).asfreq().index
    if len(grid) > _MAX_RESULT_ROWS:
        raise ValueError(f"Upsampling to {args.frequency} would create more than {_MAX_RESULT_ROWS:,} rows; choose a coarser step")
    limit = args.fill_limit
    if args.fill == UpsampleFill.none:
        out = frame.reindex(grid)
    elif args.fill != UpsampleFill.interpolate:
        out = frame.reindex(grid, method=args.fill.value, limit=limit)
    else:
        out = pd.DataFrame(index=grid)
        if numeric:
            merged = frame[numeric].reindex(frame.index.union(grid)).astype(float)
            out[numeric] = merged.interpolate(method="time", limit=limit, limit_area="inside").loc[grid]
        if other:
            out[other] = frame[other].reindex(grid, method="ffill", limit=limit)
    if limit and args.fill != UpsampleFill.none:
        described += f" (at most {limit} step(s) in a row)"
    return out[[*numeric, *other]], described


async def resample_time_series(
    dataset_version_id: str,
    time_column: str,
    frequency: str,
    mode: str = "downsample",
    aggregation: str = "mean",
    fill: str = "interpolate",
    fill_limit: Optional[int] = None,
    columns: Optional[list[str]] = None,
    drop_empty: bool = True,
    *,
    session: Optional[AsyncSession] = None,
) -> tuple[str, str]:
    """Put the rows on a regular time grid: downsample (aggregate) or upsample (fill)."""
    args = ResampleTimeSeriesArgs(
        dataset_version_id=dataset_version_id, time_column=time_column, frequency=frequency,
        mode=mode, aggregation=aggregation, fill=fill, fill_limit=fill_limit,  # type: ignore[arg-type]
        columns=columns, drop_empty=drop_empty,
    )

    async def _run(db: AsyncSession) -> tuple[str, str]:
        parent = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(parent)
        keep = [c for c in (args.columns or list(df.columns)) if c != args.time_column]
        _assert_columns_exist(df, [args.time_column, *keep])
        if not keep:
            raise ValueError("Choose at least one column besides the time column")
        offset = _offset(args.frequency)
        stamps = _timestamps(df[args.time_column], args.time_column)
        valid = stamps.notna()
        if valid.sum() < 2:
            raise ValueError(f"'{args.time_column}' needs at least two rows with readable timestamps")
        frame = df.loc[valid, keep].copy()
        frame.index = pd.DatetimeIndex(stamps[valid], name=args.time_column)
        frame = frame.sort_index(kind="stable")
        numeric = [c for c in keep if _is_plain_numeric(frame[c])]
        other = [c for c in keep if c not in numeric]

        gaps = frame.index.to_series().diff()
        gaps = gaps[gaps > pd.Timedelta(0)]
        spacing = gaps.median() if len(gaps) else None
        step = _fixed_step(offset)
        if spacing is not None and step is not None:
            if args.mode == ResampleMode.downsample and step < spacing:
                raise ValueError(
                    f"The rows are about {_step_text(spacing)} apart, so a {args.frequency} step is finer; "
                    f"that is upsampling (mode='upsample'), or choose a step of at least {_step_text(spacing)}"
                )
            if args.mode == ResampleMode.upsample and step > spacing:
                raise ValueError(
                    f"The rows are about {_step_text(spacing)} apart, so a {args.frequency} step is coarser; "
                    f"that is downsampling (mode='downsample'), or choose a step of at most {_step_text(spacing)}"
                )

        if args.mode == ResampleMode.downsample:
            out, described = _downsample(frame, offset, args, numeric, other)
        else:
            out, described = _upsample(frame, offset, args, numeric, other)
        if out.empty:
            raise ValueError("Resampling left no rows")
        result = out.reset_index()
        result = result[[c for c in df.columns if c == args.time_column or c in keep]]

        spacing_text = f" (about {_step_text(spacing)} apart)" if spacing is not None else ""
        verb = "Downsampled" if args.mode == ResampleMode.downsample else "Upsampled"
        summary = f"{verb} {len(frame)} rows{spacing_text} to {len(result)} rows every {args.frequency}: {described}."
        if (~valid).any():
            summary += f" {int((~valid).sum())} row(s) without a readable timestamp were left out."
        missing = int(result[numeric].isna().sum().sum()) if numeric else 0
        if missing:
            summary += f" {missing} numeric value(s) are missing in the result."
        return await _persist_new_version(
            db,
            parent=parent,
            df=result,
            action_performed="resample_time_series",
            parameters_used=args.model_dump(mode="json", exclude={"dataset_version_id"}, exclude_none=True),
            summary_message=summary,
        )

    return await _run_with(session, _run)


# ---------------------------------------------------------------------------
# Model-based feature selection
# ---------------------------------------------------------------------------


class ModelSelectionMethod(str, Enum):
    tree_importance = "tree_importance"
    permutation = "permutation"
    shap = "shap"


class SelectorEstimator(str, Enum):
    random_forest = "random_forest"
    extra_trees = "extra_trees"
    linear = "linear"
    ridge = "ridge"


_TREE_ESTIMATORS = frozenset({SelectorEstimator.random_forest, SelectorEstimator.extra_trees})
_MAX_SELECTION_ROWS = 20000


class SelectFeaturesModelArgs(BaseModel):
    dataset_version_id: str = Field(..., min_length=36, max_length=36)
    target_column: str = Field(..., description="Name of the target/response column.", min_length=1)
    feature_columns: Optional[list[str]] = Field(
        None,
        description="Candidate feature columns. Omit to consider every numeric column except the target.",
    )
    n_features: int = Field(5, description="Number of most important features to keep.", ge=1)
    method: ModelSelectionMethod = Field(
        ModelSelectionMethod.tree_importance,
        description=(
            "tree_importance (impurity importance of a random forest / extra trees), permutation "
            "(drop in R² on held-out rows when a feature is shuffled; any estimator) or shap (mean "
            "|SHAP value| of a linear or ridge model, computed exactly)."
        ),
    )
    estimator: Optional[SelectorEstimator] = Field(
        None,
        description="Model scored: random_forest, extra_trees, linear or ridge. Omit for random_forest (shap: linear).",
    )
    n_estimators: int = Field(200, description="Trees in random_forest / extra_trees.", ge=10, le=1000)
    alpha: float = Field(1.0, description="Regularisation strength of the ridge estimator.", gt=0)
    n_repeats: int = Field(5, description="For permutation: shuffles per feature.", ge=1, le=50)
    test_size: float = Field(0.25, description="For permutation: share of rows held out for scoring.", gt=0.05, lt=0.5)
    random_state: int = Field(42, description="Random seed.")

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        return _validate_uuid_str(value)

    @field_validator("feature_columns")
    @classmethod
    def _optional(cls, value: Optional[list[str]]) -> Optional[list[str]]:
        return _clean_optional_names(value)

    @field_validator("estimator", mode="before")
    @classmethod
    def _blank_estimator(cls, value: Any) -> Any:
        return _blank(value)

    @model_validator(mode="after")
    def _compatible(self) -> "SelectFeaturesModelArgs":
        if self.estimator is None:
            self.estimator = SelectorEstimator.linear if self.method == ModelSelectionMethod.shap else SelectorEstimator.random_forest
        if self.method == ModelSelectionMethod.tree_importance and self.estimator not in _TREE_ESTIMATORS:
            raise ValueError("tree_importance needs estimator random_forest or extra_trees")
        if self.method == ModelSelectionMethod.shap and self.estimator in _TREE_ESTIMATORS:
            raise ValueError("shap is computed exactly for linear and ridge estimators; use permutation for tree models")
        return self


def _selector_model(args: SelectFeaturesModelArgs) -> Any:
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression, Ridge

    if args.estimator == SelectorEstimator.random_forest:
        return RandomForestRegressor(n_estimators=args.n_estimators, random_state=args.random_state, n_jobs=1)
    if args.estimator == SelectorEstimator.extra_trees:
        return ExtraTreesRegressor(n_estimators=args.n_estimators, random_state=args.random_state, n_jobs=1)
    if args.estimator == SelectorEstimator.ridge:
        return Ridge(alpha=args.alpha)
    return LinearRegression()


def _importances(X: pd.DataFrame, y: pd.Series, args: SelectFeaturesModelArgs) -> tuple[np.ndarray, str]:
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split

    model = _selector_model(args)
    if args.method == ModelSelectionMethod.tree_importance:
        model.fit(X, y)
        return np.asarray(model.feature_importances_, dtype=float), "impurity importance"
    if args.method == ModelSelectionMethod.permutation:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)
        if len(X_test) < 2:
            raise ValueError("permutation needs at least 2 held-out rows; increase test_size or add rows")
        model.fit(X_train, y_train)
        scores = permutation_importance(
            model, X_test, y_test, n_repeats=args.n_repeats, random_state=args.random_state, scoring="r2",
        )
        return np.asarray(scores.importances_mean, dtype=float), f"mean drop in held-out R² over {args.n_repeats} shuffle(s)"
    model.fit(X, y)
    shap_values = (X - X.mean()).to_numpy() * np.asarray(model.coef_, dtype=float)
    return np.abs(shap_values).mean(axis=0), "mean |SHAP value| (exact for a linear model)"


async def select_features_model(
    dataset_version_id: str,
    target_column: str,
    feature_columns: Optional[list[str]] = None,
    n_features: int = 5,
    method: str = "tree_importance",
    estimator: Optional[str] = None,
    n_estimators: int = 200,
    alpha: float = 1.0,
    n_repeats: int = 5,
    test_size: float = 0.25,
    random_state: int = 42,
    *,
    session: Optional[AsyncSession] = None,
) -> tuple[str, str]:
    """Keep the n_features a fitted model finds most important (tree, permutation or SHAP importance)."""
    args = SelectFeaturesModelArgs(
        dataset_version_id=dataset_version_id, target_column=target_column, feature_columns=feature_columns,
        n_features=n_features, method=method, estimator=estimator,  # type: ignore[arg-type]
        n_estimators=n_estimators, alpha=alpha, n_repeats=n_repeats, test_size=test_size,
        random_state=random_state,
    )

    async def _run(db: AsyncSession) -> tuple[str, str]:
        parent = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(parent)
        _assert_numeric_columns(df, [args.target_column])
        if args.feature_columns:
            features = [c for c in args.feature_columns if c != args.target_column]
            _assert_numeric_columns(df, features)
        else:
            features = [
                c for c in df.columns
                if c != args.target_column and _is_plain_numeric(df[c]) and not c.startswith("Predicted_")
            ]
        if not features:
            raise ValueError("No numeric feature columns to score")
        complete = df[[*features, args.target_column]].dropna()
        if len(complete) < 10:
            raise ValueError(f"Need at least 10 rows without missing values; found {len(complete)}")
        sampled = ""
        if len(complete) > _MAX_SELECTION_ROWS:
            complete = complete.sample(n=_MAX_SELECTION_ROWS, random_state=args.random_state)
            sampled = f" (scored on a random {_MAX_SELECTION_ROWS} of the complete rows)"
        X = complete[features].astype(float)
        y = complete[args.target_column].astype(float)
        scores, described = _importances(X, y, args)
        scores = np.nan_to_num(scores, nan=0.0)
        order = np.argsort(-scores, kind="stable")
        ranked = {features[i]: round(float(scores[i]), 6) for i in order}
        ranking = ", ".join(f"{name} {score:.4g}" for name, score in list(ranked.items())[:15])
        label = f"{args.method.value} with {args.estimator.value}"
        if len(features) <= args.n_features:
            return str(parent.id), (
                f"{label} ranking ({described}){sampled}: {ranking}. Only {len(features)} feature(s), "
                f"not more than n_features={args.n_features}; nothing to drop. No changes made."
            )
        selected = [features[i] for i in order[: args.n_features]]
        dropped = [c for c in features if c not in selected]
        return await _persist_new_version(
            db,
            parent=parent,
            df=df.drop(columns=dropped),
            action_performed="select_features_model",
            parameters_used={
                **args.model_dump(mode="json", exclude={"dataset_version_id"}, exclude_none=True),
                "scores": ranked, "selected_features": selected, "dropped_features": dropped,
            },
            summary_message=(
                f"{label} kept {len(selected)} feature(s): {', '.join(selected)}; dropped {len(dropped)}: "
                f"{', '.join(dropped)}. Scores ({described}){sampled}: {ranking}."
            ),
        )

    return await _run_with(session, _run)


LLM_TOOL_SCHEMAS: list[dict[str, Any]] = [
    _openai_tool_schema(
        name="sort_data",
        description="Reorder the rows by one or more columns (sort_values), ascending or descending, missing values first or last.",
        args_model=SortDataArgs,
    ),
    _openai_tool_schema(
        name="reshape_data",
        description="Reshape between wide and long layouts: melt, stack, pivot (with aggregation) or unstack.",
        args_model=ReshapeDataArgs,
    ),
    _openai_tool_schema(
        name="combine_datasets",
        description=(
            "Combine the current dataset with another uploaded dataset of the user: merge on key "
            "columns (inner/left/right/outer), join side by side by row position, or concat the rows."
        ),
        args_model=CombineDatasetsArgs,
    ),
    _openai_tool_schema(
        name="list_datasets",
        description="List the user's uploaded datasets (file name, rows, columns) that combine_datasets can use.",
        args_model=ListDatasetsArgs,
    ),
    _openai_tool_schema(
        name="mask_values",
        description=(
            "Replace values where a condition holds (pandas mask) or where it does not (pandas "
            "where), with a fixed value or missing."
        ),
        args_model=MaskValuesArgs,
    ),
    _openai_tool_schema(
        name="map_values",
        description="Map values of columns through an old → new lookup table (Series.map / replace), e.g. recode categories.",
        args_model=MapValuesArgs,
    ),
    _openai_tool_schema(
        name="apply_function",
        description=(
            "Apply a function to numeric columns: element-wise (abs, round, sqrt, log, exp, …), "
            "cumulative (cumsum, cummax, …), rank, zscore / minmax scaling, or group statistics "
            "broadcast to every row with group_by (groupby().transform())."
        ),
        args_model=ApplyFunctionArgs,
    ),
    _openai_tool_schema(
        name="resample_time_series",
        description=(
            "Resample the rows onto a regular time grid by a timestamp column (pandas resample): "
            "downsample to a coarser step by aggregating each interval (mean, median, sum, min, max, "
            "first, last, std, count), or upsample to a finer step and fill the new time steps "
            "(time interpolation, forward/backward fill, nearest or none), e.g. to align 1-min "
            "historian data with hourly lab results. Saves a new dataset version."
        ),
        args_model=ResampleTimeSeriesArgs,
    ),
    _openai_tool_schema(
        name="select_features_model",
        description=(
            "Keep the most important features according to a fitted model: tree-based impurity "
            "importance, permutation importance on held-out rows, or exact SHAP values of a linear model."
        ),
        args_model=SelectFeaturesModelArgs,
    ),
]

TOOL_FUNCTIONS = {
    "sort_data": sort_data,
    "reshape_data": reshape_data,
    "combine_datasets": combine_datasets,
    "list_datasets": list_datasets,
    "mask_values": mask_values,
    "map_values": map_values,
    "apply_function": apply_function,
    "resample_time_series": resample_time_series,
    "select_features_model": select_features_model,
}

__all__ = ["LLM_TOOL_SCHEMAS", "TOOL_FUNCTIONS", *TOOL_FUNCTIONS, "owner_datasets", "resolve_other_dataset"]
