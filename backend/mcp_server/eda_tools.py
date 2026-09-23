"""Deterministic EDA / preprocessing tools for Groq OpenAI-compatible tool calling.

Every mutating tool loads a parquet snapshot by ``dataset_version_id``, writes a
new ``{owner}/{dataset}/{uuid}.parquet`` object to storage, logs parent-child lineage,
and returns ``(new_version_id, summary_message)``.
"""

from __future__ import annotations

import json
import re
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pydantic import BaseModel, Field, field_validator
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer
import scipy.stats.mstats as mstats
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import AsyncSessionLocal
from backend.models import Dataset, DatasetVersion
from backend.services.lineage_service import (
    LineageService,
    build_parquet_path,
)
from backend.storage import get_storage

# ---------------------------------------------------------------------------
# Input schemas (strict) — source of truth for LLM_TOOL_SCHEMAS
# ---------------------------------------------------------------------------


class MissingStrategy(str, Enum):
    drop_rows = "drop_rows"
    fill_mean = "fill_mean"
    fill_median = "fill_median"
    fill_mode = "fill_mode"
    forward_fill = "forward_fill"
    backward_fill = "backward_fill"
    knn = "knn"
    iterative = "iterative"


class OutlierMethod(str, Enum):
    z_score = "z_score"
    iqr = "iqr"
    winsorize = "winsorize"


class NormalizeStrategy(str, Enum):
    standard = "standard"
    min_max = "min_max"
    robust = "robust"
    maxabs = "maxabs"


class PlotType(str, Enum):
    scatter = "scatter"
    time_series = "time_series"
    histogram = "histogram"
    correlation_heatmap = "correlation_heatmap"
    box_plot = "box_plot"
    violin_plot = "violin_plot"
    pair_plot = "pair_plot"
    scatter_3d = "scatter_3d"


class RunDuckDBQueryArgs(BaseModel):
    """Arguments for read-only DuckDB aggregation / summary queries."""

    dataset_version_id: str = Field(
        ...,
        description="UUID of the dataset version whose parquet file to query.",
        min_length=36,
        max_length=36,
    )
    sql_query: str = Field(
        ...,
        description=(
            "Read-only SQL against the in-memory table named `dataset`. "
            "Only SELECT / WITH queries are allowed."
        ),
        min_length=1,
    )

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        return _validate_uuid_str(value)


class RemoveMissingDataArgs(BaseModel):
    """Arguments for missing-value preprocessing."""

    dataset_version_id: str = Field(
        ...,
        description="UUID of the parent dataset version to transform.",
        min_length=36,
        max_length=36,
    )
    strategy: MissingStrategy = Field(
        ...,
        description="Technique to handle missing data (e.g., drop_rows, knn, iterative).",
    )
    flag_missing: bool = Field(
        default=False,
        description="If True, creates boolean indicator columns (e.g. col_is_missing) before imputing.",
    )

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        return _validate_uuid_str(value)


class RemoveOutliersArgs(BaseModel):
    """Arguments for outlier removal on selected numeric columns."""

    dataset_version_id: str = Field(
        ...,
        description="UUID of the parent dataset version to transform.",
        min_length=36,
        max_length=36,
    )
    columns: list[str] = Field(
        ...,
        description="Numeric column names to evaluate for outliers.",
        min_length=1,
    )
    method: OutlierMethod = Field(
        ...,
        description="Outlier detection method: z_score or iqr.",
    )
    threshold: float = Field(
        ...,
        description=(
            "For z_score: absolute z-score cutoff. "
            "For iqr: multiplier on the IQR fence (commonly 1.5)."
        ),
        gt=0,
    )

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        return _validate_uuid_str(value)

    @field_validator("columns")
    @classmethod
    def _non_empty_names(cls, value: list[str]) -> list[str]:
        cleaned = [c.strip() for c in value if c and c.strip()]
        if not cleaned:
            raise ValueError("columns must contain at least one non-empty name")
        return cleaned


class NormalizeDataArgs(BaseModel):
    """Arguments for column-wise scaling with scikit-learn."""

    dataset_version_id: str = Field(
        ...,
        description="UUID of the parent dataset version to transform.",
        min_length=36,
        max_length=36,
    )
    columns: list[str] = Field(
        ...,
        description="Numeric columns to normalize.",
        min_length=1,
    )
    strategy: NormalizeStrategy = Field(
        ...,
        description="Scaling strategy: standard, min_max, or robust.",
    )

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        return _validate_uuid_str(value)

    @field_validator("columns")
    @classmethod
    def _non_empty_names(cls, value: list[str]) -> list[str]:
        cleaned = [c.strip() for c in value if c and c.strip()]
        if not cleaned:
            raise ValueError("columns must contain at least one non-empty name")
        return cleaned


class DropColumnsArgs(BaseModel):
    """Arguments for dropping columns."""

    dataset_version_id: str = Field(
        ...,
        description="UUID of the dataset version.",
        min_length=36,
        max_length=36,
    )
    columns: list[str] = Field(
        ...,
        description="List of column names to drop.",
        min_length=1,
    )

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        return _validate_uuid_str(value)

    @field_validator("columns")
    @classmethod
    def _non_empty_names(cls, value: list[str]) -> list[str]:
        cleaned = [c.strip() for c in value if c and c.strip()]
        if not cleaned:
            raise ValueError("columns must contain at least one non-empty name")
        return cleaned


class ExecuteFormulaArgs(BaseModel):
    """Arguments for executing a pandas formula to create a new column."""

    dataset_version_id: str = Field(
        ...,
        description="UUID of the dataset version.",
        min_length=36,
        max_length=36,
    )
    new_column_name: str = Field(
        ...,
        description="Name of the new column to be created.",
        min_length=1,
    )
    formula: str = Field(
        ...,
        description="A pandas eval-compatible formula using existing column names, e.g., 'A * B + C'.",
        min_length=1,
    )

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        return _validate_uuid_str(value)


class RenameColumnsArgs(BaseModel):
    """Arguments for renaming columns."""

    dataset_version_id: str = Field(
        ...,
        description="UUID of the dataset version.",
        min_length=36,
        max_length=36,
    )
    column_mapping: dict[str, str] = Field(
        ...,
        description="Dictionary mapping existing column names to new column names.",
    )

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        return _validate_uuid_str(value)

    @field_validator("column_mapping")
    @classmethod
    def _non_empty_mapping(cls, value: dict[str, str]) -> dict[str, str]:
        if not value:
            raise ValueError("column_mapping cannot be empty")
        return value


# ---------------------------------------------------------------------------
# Time-series tools
# ---------------------------------------------------------------------------

class RollingAggregateArgs(BaseModel):
    """Arguments for computing rolling window aggregates."""

    dataset_version_id: str = Field(..., description="UUID of the dataset version.", min_length=36, max_length=36)
    columns: list[str] = Field(..., description="Numeric columns to compute rolling aggregates for.", min_length=1)
    window: int = Field(..., description="Rolling window size (number of rows).", ge=2)
    agg_func: str = Field("mean", description="Aggregation function: mean, std, min, max, or sum.")
    min_periods: int = Field(1, description="Minimum number of observations required to produce a value.", ge=1)

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, v: str) -> str: return _validate_uuid_str(v)

    @field_validator("agg_func")
    @classmethod
    def _valid_agg(cls, v: str) -> str:
        allowed = {"mean", "std", "min", "max", "sum"}
        if v not in allowed:
            raise ValueError(f"agg_func must be one of {allowed}")
        return v


class CreateLagFeaturesArgs(BaseModel):
    """Arguments for creating lag/shift features."""

    dataset_version_id: str = Field(..., description="UUID of the dataset version.", min_length=36, max_length=36)
    columns: list[str] = Field(..., description="Columns to create lag features for.", min_length=1)
    lag_periods: list[int] = Field(..., description="List of lag periods (positive integers, e.g. [1, 5, 10]).", min_length=1)

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, v: str) -> str: return _validate_uuid_str(v)

    @field_validator("lag_periods")
    @classmethod
    def _valid_lags(cls, v: list[int]) -> list[int]:
        if any(p <= 0 for p in v):
            raise ValueError("All lag_periods must be positive integers")
        return v


# ---------------------------------------------------------------------------
# Advanced feature selection & data quality tools
# ---------------------------------------------------------------------------

class SelectFeaturesRFEArgs(BaseModel):
    """Arguments for Recursive Feature Elimination."""

    dataset_version_id: str = Field(..., description="UUID of the dataset version.", min_length=36, max_length=36)
    target_column: str = Field(..., description="Name of the target/response column.")
    n_features_to_select: int = Field(5, description="Number of top features to keep.", ge=1)
    estimator: str = Field("ridge", description="Base estimator for RFE: 'ridge' or 'lasso'.")

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, v: str) -> str: return _validate_uuid_str(v)

    @field_validator("estimator")
    @classmethod
    def _valid_estimator(cls, v: str) -> str:
        if v not in {"ridge", "lasso"}:
            raise ValueError("estimator must be 'ridge' or 'lasso'")
        return v


class DetectAnomaliesArgs(BaseModel):
    """Arguments for Isolation Forest anomaly detection."""

    dataset_version_id: str = Field(..., description="UUID of the dataset version.", min_length=36, max_length=36)
    contamination: float = Field(0.05, description="Expected proportion of anomalies in the dataset (0.0–0.5).", ge=0.01, le=0.5)
    drop_anomalies: bool = Field(True, description="If True, drop anomalous rows. If False, add an 'is_anomaly' flag column instead.")

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, v: str) -> str: return _validate_uuid_str(v)


class DropCollinearFeaturesArgs(BaseModel):
    """Arguments for dropping highly-correlated (collinear) features."""

    dataset_version_id: str = Field(..., description="UUID of the dataset version.", min_length=36, max_length=36)
    threshold: float = Field(0.95, description="Pearson correlation threshold above which one of a pair is dropped.", ge=0.5, le=1.0)
    target_column: Optional[str] = Field(None, description="If provided, this column is excluded from the collinearity analysis and never dropped.")

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, v: str) -> str: return _validate_uuid_str(v)


# ---------------------------------------------------------------------------
# Statistical profiler
# ---------------------------------------------------------------------------

class GetColumnStatisticsArgs(BaseModel):
    """Arguments for computing per-column descriptive statistics."""

    dataset_version_id: str = Field(..., description="UUID of the dataset version.", min_length=36, max_length=36)
    columns: Optional[list[str]] = Field(None, description="Columns to profile. If omitted, all columns are profiled.")

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, v: str) -> str: return _validate_uuid_str(v)


class GenerateCustomPlotArgs(BaseModel):
    """Arguments for Plotly figure generation for the React frontend."""

    dataset_version_id: str = Field(
        ...,
        description="UUID of the dataset version to visualize.",
        min_length=36,
        max_length=36,
    )
    plot_type: PlotType = Field(
        ...,
        description=(
            "Figure type: scatter, time_series, histogram, correlation_heatmap, "
            "box_plot, violin_plot, pair_plot, or scatter_3d."
        ),
    )
    x_column: str = Field(
        ...,
        description="X-axis / primary column (ignored for correlation_heatmap and pair_plot).",
        min_length=1,
    )
    y_columns: list[str] = Field(
        default_factory=list,
        description=(
            "Y-axis columns for scatter/time_series/box_plot/violin_plot; "
            "columns for histogram when x_column is unused; "
            "columns included in correlation_heatmap and pair_plot."
        ),
    )
    color_column: Optional[str] = Field(
        None,
        description="Optional column to color-code points in scatter and time_series plots.",
    )
    trendline: bool = Field(
        False,
        description="If True, add an OLS trendline to scatter plots.",
    )
    z_column: Optional[str] = Field(
        None,
        description="Z-axis column for scatter_3d plots.",
    )
    max_points: int = Field(
        10000,
        description="Maximum number of data points to render. If dataset is larger, it is randomly subsampled.",
        ge=100,
        le=100000,
    )

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        return _validate_uuid_str(value)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_WRITE_SQL_PATTERN = re.compile(
    r"\b("
    r"INSERT|UPDATE|DELETE|MERGE|REPLACE|TRUNCATE|"
    r"DROP|CREATE|ALTER|ATTACH|DETACH|COPY|EXPORT|IMPORT|"
    r"INSTALL|LOAD|PRAGMA|CALL|EXECUTE|VACUUM|CHECKPOINT|"
    r"GRANT|REVOKE|SET|RESET|USE|PREPARE|DEALLOCATE"
    r")\b",
    re.IGNORECASE,
)


def _validate_uuid_str(value: str) -> str:
    try:
        return str(uuid.UUID(value))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError(f"Invalid UUID: {value!r}") from exc


def _resolve_storage_path(file_path: str) -> Path:
    """Local path of a stored version (downloaded first on remote storage)."""
    return get_storage().fetch(file_path)


def _write_parquet(key: str, df: pd.DataFrame) -> None:
    get_storage().write(key, lambda path: df.to_parquet(path, index=False))


async def _get_version(
    session: AsyncSession,
    dataset_version_id: str,
) -> DatasetVersion:
    version = await session.get(DatasetVersion, uuid.UUID(dataset_version_id))
    if version is None:
        raise ValueError(f"Dataset version not found: {dataset_version_id}")
    return version


def _load_dataframe(version: DatasetVersion) -> pd.DataFrame:
    return pd.read_parquet(_resolve_storage_path(version.file_path))


def _assert_columns_exist(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in dataset: {missing}")


def _assert_numeric_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    _assert_columns_exist(df, columns)
    non_numeric = [
        c for c in columns if not pd.api.types.is_numeric_dtype(df[c])
    ]
    if non_numeric:
        raise ValueError(f"Columns must be numeric: {non_numeric}")


def _assert_read_only_sql(sql_query: str) -> str:
    cleaned = sql_query.strip().rstrip(";")
    if not cleaned:
        raise ValueError("sql_query must not be empty")
    if _WRITE_SQL_PATTERN.search(cleaned):
        raise ValueError(
            "Only read-only SELECT / WITH queries are allowed against `dataset`"
        )
    first = cleaned.lstrip("(").lstrip().split(None, 1)[0].upper()
    if first not in {"SELECT", "WITH", "DESCRIBE", "SHOW", "SUMMARIZE", "FROM"}:
        raise ValueError(
            "Only read-only SELECT / WITH queries are allowed against `dataset`"
        )
    return cleaned


async def _persist_new_version(
    session: AsyncSession,
    *,
    parent: DatasetVersion,
    df: pd.DataFrame,
    action_performed: str,
    parameters_used: dict[str, Any],
    summary_message: str,
) -> tuple[str, str]:
    """Write parquet, log lineage, return (new_version_id, summary_message)."""
    dataset = await session.get(Dataset, parent.dataset_id)
    if dataset is None:
        raise ValueError(f"Dataset not found: {parent.dataset_id}")
    new_id = uuid.uuid4()
    relative_path = build_parquet_path(dataset.owner_id, dataset.id, new_id)
    _write_parquet(relative_path, df)

    svc = LineageService(session)
    version = await svc.save_version(
        dataset_id=parent.dataset_id,
        action_performed=action_performed,
        parameters_used=parameters_used,
        parent_version_id=parent.id,
        file_path=relative_path,
        version_id=new_id,
    )
    await session.commit()
    return str(version.id), summary_message


def _rows_to_jsonable(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert DuckDB / pandas scalars into JSON-serializable values."""
    out: list[dict[str, Any]] = []
    for row in rows:
        converted: dict[str, Any] = {}
        for key, value in row.items():
            if value is None or isinstance(value, (str, int, float, bool)):
                converted[key] = value
            elif isinstance(value, (np.integer,)):
                converted[key] = int(value)
            elif isinstance(value, (np.floating,)):
                converted[key] = float(value) if np.isfinite(value) else None
            elif isinstance(value, (np.bool_,)):
                converted[key] = bool(value)
            elif pd.isna(value):
                converted[key] = None
            else:
                converted[key] = str(value)
        out.append(converted)
    return out


_FAKE_NULL_LITERALS = (
    "",
    "NA",
    "N/A",
    "null",
    "NaN",
    "nan",
    "None",
    "?",
)

# Empty / whitespace-only strings (applied after strip for object columns).
_WHITESPACE_ONLY_PATTERN = re.compile(r"^\s*$")


def _coerce_fake_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace CSV 'fake' nulls with ``numpy.nan`` so ``dropna`` / fillna work.

    Coerces empty strings, pure whitespace, and common text null literals
    (NA, N/A, null, NaN, nan, None, ?), case-insensitively for literals.
    """
    result = df.copy()
    literal_set = {lit.casefold() for lit in _FAKE_NULL_LITERALS if lit}

    for col in result.columns:
        series = result[col]
        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            continue

        as_str = series.map(
            lambda v: v.strip() if isinstance(v, str) else v
        )

        mask_whitespace = series.map(
            lambda v: isinstance(v, str) and bool(_WHITESPACE_ONLY_PATTERN.match(v))
        )
        mask_empty = as_str.map(lambda v: isinstance(v, str) and v == "")
        mask_literals = as_str.map(
            lambda v: isinstance(v, str) and v.casefold() in literal_set
        )
        # Also catch exact list replacements before strip (e.g. literal "").
        mask_exact = series.isin(list(_FAKE_NULL_LITERALS))

        fake_mask = mask_whitespace | mask_empty | mask_literals | mask_exact
        if fake_mask.any():
            result.loc[fake_mask, col] = np.nan

    # Standard list replacement on object columns (belt-and-suspenders).
    result.replace(list(_FAKE_NULL_LITERALS), np.nan, inplace=True)
    return result


def _openai_tool_schema(
    *,
    name: str,
    description: str,
    args_model: type[BaseModel],
) -> dict[str, Any]:
    schema = args_model.model_json_schema()
    # OpenAI / Groq expect a flat JSON Schema object under ``parameters``.
    schema.pop("title", None)
    
    # Do not expose dataset_version_id to the LLM; it is injected by execution_agent.
    # Exposing it causes the LLM to hallucinate values and fail Groq strict validation.
    if "properties" in schema and "dataset_version_id" in schema["properties"]:
        del schema["properties"]["dataset_version_id"]
    if "required" in schema and "dataset_version_id" in schema["required"]:
        schema["required"].remove("dataset_version_id")

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": schema,
        },
    }


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


async def run_duckdb_query(
    dataset_version_id: str,
    sql_query: str,
    *,
    session: Optional[AsyncSession] = None,
) -> dict[str, Any]:
    """
    Load a dataset version into an in-memory DuckDB table named ``dataset``
    and execute a read-only aggregation/summary query. Returns JSON rows.
    """
    args = RunDuckDBQueryArgs(
        dataset_version_id=dataset_version_id,
        sql_query=sql_query,
    )
    safe_sql = _assert_read_only_sql(args.sql_query)

    async def _run(db: AsyncSession) -> dict[str, Any]:
        version = await _get_version(db, args.dataset_version_id)
        path = _resolve_storage_path(version.file_path)

        con = duckdb.connect(database=":memory:")
        try:
            con.execute(
                "CREATE TABLE dataset AS SELECT * FROM read_parquet(?)",
                [str(path)],
            )
            # The SQL comes from the model, i.e. from user prompts: once the
            # table is loaded it must not reach files, URLs or extensions.
            con.execute("SET enable_external_access = false")
            con.execute("SET lock_configuration = true")
            relation = con.execute(safe_sql)
            columns = [desc[0] for desc in relation.description]
            records = [dict(zip(columns, row)) for row in relation.fetchall()]
            return {
                "dataset_version_id": args.dataset_version_id,
                "row_count": len(records),
                "columns": columns,
                "rows": _rows_to_jsonable(records),
            }
        finally:
            con.close()

    if session is not None:
        return await _run(session)
    async with AsyncSessionLocal() as db:
        return await _run(db)


async def drop_columns(
    dataset_version_id: str,
    columns: list[str],
    *,
    session: Optional[AsyncSession] = None,
) -> tuple[str, str]:
    """
    Drop specified columns from the dataset.

    Returns:
        (new_version_id, summary_message)
    """
    args = DropColumnsArgs(
        dataset_version_id=dataset_version_id,
        columns=columns,
    )

    async def _run(db: AsyncSession) -> tuple[str, str]:
        parent = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(parent)
        
        cols_to_drop = [c for c in args.columns if c in df.columns]
        if not cols_to_drop:
            return str(parent.id), "No matching columns found to drop."
            
        df = df.drop(columns=cols_to_drop)
        
        return await _persist_new_version(
            db,
            parent=parent,
            df=df,
            action_performed="drop_columns",
            parameters_used={
                "columns_dropped": cols_to_drop,
            },
            summary_message=f"Dropped columns: {', '.join(cols_to_drop)}",
        )

    if session is not None:
        return await _run(session)
    async with AsyncSessionLocal() as db:
        return await _run(db)


async def execute_formula(
    dataset_version_id: str,
    new_column_name: str,
    formula: str,
    *,
    session: Optional[AsyncSession] = None,
) -> tuple[str, str]:
    """
    Create a new column by evaluating a mathematical formula over existing columns.

    Returns:
        (new_version_id, summary_message)
    """
    args = ExecuteFormulaArgs(
        dataset_version_id=dataset_version_id,
        new_column_name=new_column_name,
        formula=formula,
    )

    async def _run(db: AsyncSession) -> tuple[str, str]:
        parent = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(parent)
        
        try:
            # Safely evaluate the formula. We use engine='python' or just the default.
            new_col_series = df.eval(args.formula)
            df[args.new_column_name] = new_col_series
        except Exception as e:
            return str(parent.id), f"Failed to execute formula: {e}"
        
        return await _persist_new_version(
            db,
            parent=parent,
            df=df,
            action_performed="execute_formula",
            parameters_used={
                "new_column_name": args.new_column_name,
                "formula": args.formula,
            },
            summary_message=f"Created new column '{args.new_column_name}' using formula: {args.formula}",
        )

    if session is not None:
        return await _run(session)
    async with AsyncSessionLocal() as db:
        return await _run(db)


async def rename_columns(
    dataset_version_id: str,
    column_mapping: dict[str, str],
    *,
    session: Optional[AsyncSession] = None,
) -> tuple[str, str]:
    """
    Rename columns in the dataset.

    Returns:
        (new_version_id, summary_message)
    """
    args = RenameColumnsArgs(
        dataset_version_id=dataset_version_id,
        column_mapping=column_mapping,
    )

    async def _run(db: AsyncSession) -> tuple[str, str]:
        parent = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(parent)
        
        # Only rename columns that actually exist
        valid_mapping = {k: v for k, v in args.column_mapping.items() if k in df.columns}
        if not valid_mapping:
            return str(parent.id), "No matching columns found to rename."
            
        df = df.rename(columns=valid_mapping)
        
        # Format a nice message
        renamed_str = ", ".join(f"'{k}' -> '{v}'" for k, v in valid_mapping.items())
        
        return await _persist_new_version(
            db,
            parent=parent,
            df=df,
            action_performed="rename_columns",
            parameters_used={
                "column_mapping": valid_mapping,
            },
            summary_message=f"Renamed columns: {renamed_str}",
        )

    if session is not None:
        return await _run(session)
    async with AsyncSessionLocal() as db:
        return await _run(db)




# ---------------------------------------------------------------------------
# Time-series tool implementations
# ---------------------------------------------------------------------------


async def rolling_aggregate(
    dataset_version_id: str,
    columns: list[str],
    window: int,
    agg_func: str = "mean",
    min_periods: int = 1,
    *,
    session: Optional[AsyncSession] = None,
) -> tuple[str, str]:
    """Compute rolling window aggregates and add as new columns.

    Returns:
        (new_version_id, summary_message)
    """
    args = RollingAggregateArgs(
        dataset_version_id=dataset_version_id,
        columns=columns,
        window=window,
        agg_func=agg_func,
        min_periods=min_periods,
    )

    async def _run(db: AsyncSession) -> tuple[str, str]:
        parent = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(parent)
        _assert_numeric_columns(df, args.columns)

        result = df.copy()
        for col in args.columns:
            new_col_name = f"{col}_rolling{args.window}_{args.agg_func}"
            roller = result[col].rolling(window=args.window, min_periods=args.min_periods)
            result[new_col_name] = getattr(roller, args.agg_func)()

        created = [f"{c}_rolling{args.window}_{args.agg_func}" for c in args.columns]
        return await _persist_new_version(
            db,
            parent=parent,
            df=result,
            action_performed="rolling_aggregate",
            parameters_used={"columns": args.columns, "window": args.window, "agg_func": args.agg_func},
            summary_message=f"Created {len(created)} rolling {args.agg_func}(window={args.window}) column(s): {', '.join(created)}",
        )

    if session is not None:
        return await _run(session)
    async with AsyncSessionLocal() as db:
        return await _run(db)


async def create_lag_features(
    dataset_version_id: str,
    columns: list[str],
    lag_periods: list[int],
    *,
    session: Optional[AsyncSession] = None,
) -> tuple[str, str]:
    """Create lagged versions of columns for time-series modeling.

    Returns:
        (new_version_id, summary_message)
    """
    args = CreateLagFeaturesArgs(
        dataset_version_id=dataset_version_id,
        columns=columns,
        lag_periods=lag_periods,
    )

    async def _run(db: AsyncSession) -> tuple[str, str]:
        parent = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(parent)
        _assert_columns_exist(df, args.columns)

        result = df.copy()
        created: list[str] = []
        for col in args.columns:
            for lag in args.lag_periods:
                new_col_name = f"{col}_lag{lag}"
                result[new_col_name] = result[col].shift(lag)
                created.append(new_col_name)

        return await _persist_new_version(
            db,
            parent=parent,
            df=result,
            action_performed="create_lag_features",
            parameters_used={"columns": args.columns, "lag_periods": args.lag_periods},
            summary_message=f"Created {len(created)} lag feature column(s) for lags {args.lag_periods}.",
        )

    if session is not None:
        return await _run(session)
    async with AsyncSessionLocal() as db:
        return await _run(db)


# ---------------------------------------------------------------------------
# Advanced feature selection & data quality tool implementations
# ---------------------------------------------------------------------------


async def select_features_rfe(
    dataset_version_id: str,
    target_column: str,
    n_features_to_select: int = 5,
    estimator: str = "ridge",
    *,
    session: Optional[AsyncSession] = None,
) -> tuple[str, str]:
    """Drop low-importance features using Recursive Feature Elimination (RFE).

    Returns:
        (new_version_id, summary_message)
    """
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import Ridge, Lasso

    args = SelectFeaturesRFEArgs(
        dataset_version_id=dataset_version_id,
        target_column=target_column,
        n_features_to_select=n_features_to_select,
        estimator=estimator,
    )

    async def _run(db: AsyncSession) -> tuple[str, str]:
        parent = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(parent)
        _assert_columns_exist(df, [args.target_column])

        feature_cols = [
            c for c in df.columns
            if c != args.target_column and pd.api.types.is_numeric_dtype(df[c])
        ]
        if len(feature_cols) < args.n_features_to_select:
            return str(parent.id), (
                f"Not enough numeric feature columns ({len(feature_cols)}) to select {args.n_features_to_select}. "
                "No changes made."
            )

        X = df[feature_cols].fillna(0)
        y = df[args.target_column].fillna(0)

        base_est = Ridge(alpha=1.0) if args.estimator == "ridge" else Lasso(alpha=1.0)
        rfe = RFE(estimator=base_est, n_features_to_select=args.n_features_to_select)
        rfe.fit(X, y)

        selected = [feature_cols[i] for i, s in enumerate(rfe.support_) if s]
        dropped = [feature_cols[i] for i, s in enumerate(rfe.support_) if not s]

        result = df.drop(columns=dropped)
        return await _persist_new_version(
            db,
            parent=parent,
            df=result,
            action_performed="select_features_rfe",
            parameters_used={
                "target_column": args.target_column,
                "n_features_to_select": args.n_features_to_select,
                "estimator": args.estimator,
                "selected_features": selected,
                "dropped_features": dropped,
            },
            summary_message=(
                f"RFE selected {len(selected)} features: {', '.join(selected)}. "
                f"Dropped {len(dropped)} features: {', '.join(dropped)}."
            ),
        )

    if session is not None:
        return await _run(session)
    async with AsyncSessionLocal() as db:
        return await _run(db)


async def detect_anomalies(
    dataset_version_id: str,
    contamination: float = 0.05,
    drop_anomalies: bool = True,
    *,
    session: Optional[AsyncSession] = None,
) -> tuple[str, str]:
    """Detect multivariate anomalies using Isolation Forest.

    Returns:
        (new_version_id, summary_message)
    """
    from sklearn.ensemble import IsolationForest

    args = DetectAnomaliesArgs(
        dataset_version_id=dataset_version_id,
        contamination=contamination,
        drop_anomalies=drop_anomalies,
    )

    async def _run(db: AsyncSession) -> tuple[str, str]:
        parent = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(parent)

        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            return str(parent.id), "No numeric columns found for anomaly detection."

        X = df[numeric_cols].fillna(0)
        iso = IsolationForest(contamination=args.contamination, random_state=42, n_jobs=-1)
        preds = iso.fit_predict(X)  # -1 = anomaly, 1 = inlier

        anomaly_mask = preds == -1
        n_anomalies = int(anomaly_mask.sum())

        if args.drop_anomalies:
            result = df[~anomaly_mask].reset_index(drop=True)
            action_msg = f"Detected and removed {n_anomalies} anomalous rows using Isolation Forest (contamination={args.contamination})."
        else:
            result = df.copy()
            result["is_anomaly"] = anomaly_mask.astype(int)
            action_msg = f"Detected {n_anomalies} anomalous rows; added 'is_anomaly' flag column (contamination={args.contamination})."

        return await _persist_new_version(
            db,
            parent=parent,
            df=result,
            action_performed="detect_anomalies",
            parameters_used={
                "contamination": args.contamination,
                "drop_anomalies": args.drop_anomalies,
                "n_anomalies_found": n_anomalies,
            },
            summary_message=action_msg,
        )

    if session is not None:
        return await _run(session)
    async with AsyncSessionLocal() as db:
        return await _run(db)


async def drop_collinear_features(
    dataset_version_id: str,
    threshold: float = 0.95,
    target_column: Optional[str] = None,
    *,
    session: Optional[AsyncSession] = None,
) -> tuple[str, str]:
    """Drop highly-correlated feature columns that cause collinearity issues.

    For each pair of features with |corr| > threshold, the one with lower variance is dropped.

    Returns:
        (new_version_id, summary_message)
    """
    args = DropCollinearFeaturesArgs(
        dataset_version_id=dataset_version_id,
        threshold=threshold,
        target_column=target_column,
    )

    async def _run(db: AsyncSession) -> tuple[str, str]:
        parent = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(parent)

        # Build feature set: numeric columns excluding target
        feature_cols = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and c != args.target_column
        ]
        if len(feature_cols) < 2:
            return str(parent.id), "Not enough numeric columns for collinearity analysis."

        corr_matrix = df[feature_cols].corr().abs()
        variances = df[feature_cols].var()

        # Upper triangle of correlation matrix
        to_drop: set[str] = set()
        for i in range(len(feature_cols)):
            for j in range(i + 1, len(feature_cols)):
                col_i = feature_cols[i]
                col_j = feature_cols[j]
                if col_i in to_drop or col_j in to_drop:
                    continue
                if corr_matrix.loc[col_i, col_j] > args.threshold:
                    # Drop the one with lower variance
                    if variances[col_i] <= variances[col_j]:
                        to_drop.add(col_i)
                    else:
                        to_drop.add(col_j)

        if not to_drop:
            return str(parent.id), f"No column pairs found with |correlation| > {args.threshold}. No changes made."

        result = df.drop(columns=list(to_drop))
        return await _persist_new_version(
            db,
            parent=parent,
            df=result,
            action_performed="drop_collinear_features",
            parameters_used={"threshold": args.threshold, "dropped_columns": sorted(to_drop)},
            summary_message=(
                f"Dropped {len(to_drop)} collinear feature(s) with |corr| > {args.threshold}: "
                f"{', '.join(sorted(to_drop))}"
            ),
        )

    if session is not None:
        return await _run(session)
    async with AsyncSessionLocal() as db:
        return await _run(db)


# ---------------------------------------------------------------------------
# Statistical profiler tool implementation
# ---------------------------------------------------------------------------


async def get_column_statistics(
    dataset_version_id: str,
    columns: Optional[list[str]] = None,
    *,
    session: Optional[AsyncSession] = None,
) -> dict[str, Any]:
    """Compute per-column descriptive statistics: count, nulls, mean, std, quantiles.

    Returns:
        JSON-serializable dict with 'columns' list and 'statistics' per column.
    """
    args = GetColumnStatisticsArgs(
        dataset_version_id=dataset_version_id,
        columns=columns,
    )

    async def _run(db: AsyncSession) -> dict[str, Any]:
        version = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(version)

        target_cols = args.columns if args.columns else list(df.columns)
        target_cols = [c for c in target_cols if c in df.columns]

        stats: dict[str, dict[str, Any]] = {}
        for col in target_cols:
            s = df[col]
            null_count = int(s.isna().sum())
            total = len(s)
            stat: dict[str, Any] = {
                "dtype": str(s.dtype),
                "count": total,
                "null_count": null_count,
                "null_pct": round(null_count / total * 100, 2) if total > 0 else 0.0,
            }
            if pd.api.types.is_numeric_dtype(s):
                desc = s.describe(percentiles=[0.25, 0.5, 0.75])
                stat.update({
                    "mean": round(float(desc["mean"]), 6) if not np.isnan(desc["mean"]) else None,
                    "std": round(float(desc["std"]), 6) if not np.isnan(desc["std"]) else None,
                    "min": round(float(desc["min"]), 6) if not np.isnan(desc["min"]) else None,
                    "p25": round(float(desc["25%"]), 6) if not np.isnan(desc["25%"]) else None,
                    "median": round(float(desc["50%"]), 6) if not np.isnan(desc["50%"]) else None,
                    "p75": round(float(desc["75%"]), 6) if not np.isnan(desc["75%"]) else None,
                    "max": round(float(desc["max"]), 6) if not np.isnan(desc["max"]) else None,
                })
            else:
                stat.update({
                    "n_unique": int(s.nunique()),
                    "top_values": s.value_counts().head(5).to_dict(),
                })
            stats[col] = stat

        return {
            "dataset_version_id": args.dataset_version_id,
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "columns": target_cols,
            "statistics": stats,
        }

    if session is not None:
        return await _run(session)
    async with AsyncSessionLocal() as db:
        return await _run(db)


async def remove_missing_data(
    dataset_version_id: str,
    strategy: str,
    flag_missing: bool = False,
    *,
    session: Optional[AsyncSession] = None,
) -> tuple[str, str]:
    """Remove or impute missing values; save a new version and log lineage."""
    args = RemoveMissingDataArgs(
        dataset_version_id=dataset_version_id,
        strategy=strategy,  # type: ignore[arg-type]
        flag_missing=flag_missing,
    )

    async def _run(db: AsyncSession) -> tuple[str, str]:
        parent = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(parent)
        initial_rows = len(df)

        # Coerce CSV "fake" nulls (empty strings, whitespace, "NA", "N/A",
        # "null", "NaN", "nan", "None", "?") into real np.nan so that
        # dropna() and fillna() strategies can detect them.
        df = _coerce_fake_nulls(df)

        before_na = int(df.isna().sum().sum())

        if args.flag_missing:
            for col in df.columns:
                if df[col].isna().any():
                    df[f"{col}_is_missing"] = df[col].isna().astype(int)

        if args.strategy == MissingStrategy.drop_rows:
            result = df.dropna()
        elif args.strategy == MissingStrategy.fill_mean:
            result = df.copy()
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            result[numeric_cols] = result[numeric_cols].fillna(result[numeric_cols].mean())
        elif args.strategy == MissingStrategy.fill_median:
            result = df.copy()
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            result[numeric_cols] = result[numeric_cols].fillna(result[numeric_cols].median())
        elif args.strategy == MissingStrategy.fill_mode:
            result = df.copy()
            for col in result.columns:
                mode_vals = result[col].mode()
                if not mode_vals.empty:
                    result[col] = result[col].fillna(mode_vals[0])
        elif args.strategy == MissingStrategy.forward_fill:
            result = df.ffill()
        elif args.strategy == MissingStrategy.backward_fill:
            result = df.bfill()
        elif args.strategy == MissingStrategy.knn:
            result = df.copy()
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            imputer = KNNImputer()
            if len(numeric_cols) > 0:
                result[numeric_cols] = imputer.fit_transform(result[numeric_cols])
        elif args.strategy == MissingStrategy.iterative:
            result = df.copy()
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            imputer = IterativeImputer(random_state=42)
            if len(numeric_cols) > 0:
                result[numeric_cols] = imputer.fit_transform(result[numeric_cols])
        else:
            result = df.dropna()

        final_rows = len(result)
        rows_dropped = initial_rows - final_rows
        after_na = int(result.isna().sum().sum())
        summary = (
            f"Applied missing-data strategy '{args.strategy.value}': "
            f"rows {initial_rows} → {final_rows} "
            f"({rows_dropped} row(s) dropped), "
            f"null cells {before_na} → {after_na}."
        )
        return await _persist_new_version(
            db,
            parent=parent,
            df=result,
            action_performed="remove_missing_data",
            parameters_used={"strategy": args.strategy.value, "flag_missing": args.flag_missing},
            summary_message=summary,
        )

    if session is not None:
        return await _run(session)
    async with AsyncSessionLocal() as db:
        return await _run(db)


async def remove_outliers(
    dataset_version_id: str,
    columns: list[str],
    method: str,
    threshold: float,
    *,
    session: Optional[AsyncSession] = None,
) -> tuple[str, str]:
    """Drop outlier rows via z-score or IQR fences; save version + lineage."""
    args = RemoveOutliersArgs(
        dataset_version_id=dataset_version_id,
        columns=columns,
        method=method,  # type: ignore[arg-type]
        threshold=threshold,
    )

    async def _run(db: AsyncSession) -> tuple[str, str]:
        parent = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(parent)
        _assert_numeric_columns(df, args.columns)

        result = df.copy()
        if args.method == OutlierMethod.winsorize:
            # Winsorization modifies data in-place instead of dropping rows.
            # Here threshold is used as the symmetric limits, e.g., 0.05 for 5th & 95th percentiles.
            limit = float(args.threshold)
            if limit < 0 or limit >= 0.5:
                # Fallback if user passes something weird like 3.0 (from Z-score)
                limit = 0.05
            for col in args.columns:
                result[col] = mstats.winsorize(result[col], limits=[limit, limit])
            removed = 0
            summary = (
                f"Winsorized columns {args.columns} using limits "
                f"[{limit}, {1 - limit}]."
            )
        else:
            mask = pd.Series(True, index=df.index)
            if args.method == OutlierMethod.z_score:
                for col in args.columns:
                    series = df[col]
                    std = series.std(ddof=0)
                    if std == 0 or pd.isna(std):
                        continue
                    z = (series - series.mean()) / std
                    mask &= z.abs() <= args.threshold
            else:  # iqr
                for col in args.columns:
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - args.threshold * iqr
                    upper = q3 + args.threshold * iqr
                    mask &= df[col].between(lower, upper)
            
            result = df.loc[mask].copy()
            removed = len(df) - len(result)
            summary = (
                f"Removed {removed} outlier row(s) using {args.method.value} "
                f"(threshold={args.threshold}) on columns {args.columns}; "
                f"rows {len(df)} → {len(result)}."
            )
        return await _persist_new_version(
            db,
            parent=parent,
            df=result,
            action_performed="remove_outliers",
            parameters_used={
                "columns": args.columns,
                "method": args.method.value,
                "threshold": args.threshold,
            },
            summary_message=summary,
        )

    if session is not None:
        return await _run(session)
    async with AsyncSessionLocal() as db:
        return await _run(db)


async def normalize_data(
    dataset_version_id: str,
    columns: list[str],
    strategy: str,
    *,
    session: Optional[AsyncSession] = None,
) -> tuple[str, str]:
    """Scale columns with scikit-learn; save a new version and log lineage."""
    args = NormalizeDataArgs(
        dataset_version_id=dataset_version_id,
        columns=columns,
        strategy=strategy,  # type: ignore[arg-type]
    )

    scalers: dict[NormalizeStrategy, type] = {
        NormalizeStrategy.standard: StandardScaler,
        NormalizeStrategy.min_max: MinMaxScaler,
        NormalizeStrategy.robust: RobustScaler,
        NormalizeStrategy.maxabs: MaxAbsScaler,
    }

    async def _run(db: AsyncSession) -> tuple[str, str]:
        parent = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(parent)
        _assert_numeric_columns(df, args.columns)

        result = df.copy()
        scaler = scalers[args.strategy]()
        result[args.columns] = scaler.fit_transform(result[args.columns])

        summary = (
            f"Normalized columns {args.columns} with strategy "
            f"'{args.strategy.value}'."
        )
        return await _persist_new_version(
            db,
            parent=parent,
            df=result,
            action_performed="normalize_data",
            parameters_used={
                "columns": args.columns,
                "strategy": args.strategy.value,
            },
            summary_message=summary,
        )

    if session is not None:
        return await _run(session)
    async with AsyncSessionLocal() as db:
        return await _run(db)


async def generate_custom_plot(
    dataset_version_id: str,
    plot_type: str,
    x_column: str,
    y_columns: list[str],
    color_column: Optional[str] = None,
    trendline: bool = False,
    z_column: Optional[str] = None,
    max_points: int = 10000,
    *,
    session: Optional[AsyncSession] = None,
) -> dict[str, Any]:
    """
    Build a Plotly figure and return ``json.loads(fig.to_json())`` for
    ``react-plotly.js`` rendering. Does not mutate dataset lineage.

    Supports: scatter, time_series, histogram, correlation_heatmap,
              box_plot, violin_plot, pair_plot, scatter_3d.
    Large datasets are automatically subsampled to max_points rows.
    """
    args = GenerateCustomPlotArgs(
        dataset_version_id=dataset_version_id,
        plot_type=plot_type,  # type: ignore[arg-type]
        x_column=x_column,
        y_columns=y_columns,
        color_column=color_column,
        trendline=trendline,
        z_column=z_column,
        max_points=max_points,
    )

    async def _run(db: AsyncSession) -> dict[str, Any]:
        version = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(version)
        # Subsample large datasets to prevent browser crashes
        if len(df) > args.max_points:
            df = df.sample(n=args.max_points, random_state=42).reset_index(drop=True)
        fig = _build_figure(df, args)
        return json.loads(fig.to_json())

    if session is not None:
        return await _run(session)
    async with AsyncSessionLocal() as db:
        return await _run(db)


def _build_figure(df: pd.DataFrame, args: GenerateCustomPlotArgs) -> go.Figure:
    plot_type = args.plot_type
    # Resolve optional trendline mode for px.scatter
    trendline_mode = "ols" if args.trendline else None

    # ── Correlation Heatmap ─────────────────────────────────────────────────
    if plot_type == PlotType.correlation_heatmap:
        cols = args.y_columns or list(df.select_dtypes(include=[np.number]).columns)
        if not cols:
            raise ValueError("correlation_heatmap requires numeric columns")
        _assert_numeric_columns(df, cols)
        corr = df[cols].corr(numeric_only=True)
        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            title="Correlation heatmap",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
        )
        return fig

    # ── Histogram ───────────────────────────────────────────────────────────
    if plot_type == PlotType.histogram:
        target_cols = args.y_columns or [args.x_column]
        _assert_columns_exist(df, target_cols)
        if len(target_cols) == 1:
            fig = px.histogram(
                df, x=target_cols[0],
                color=args.color_column if args.color_column else None,
                title=f"Histogram of {target_cols[0]}",
            )
        else:
            melted = df[target_cols].melt(var_name="column", value_name="value")
            fig = px.histogram(
                melted, x="value", color="column",
                barmode="overlay", opacity=0.7, title="Histogram",
            )
        return fig

    # ── Scatter ─────────────────────────────────────────────────────────────
    if plot_type == PlotType.scatter:
        if not args.y_columns:
            raise ValueError("scatter requires at least one y_column")
        _assert_columns_exist(df, [args.x_column, *args.y_columns])
        if args.color_column:
            _assert_columns_exist(df, [args.color_column])
        if len(args.y_columns) == 1:
            fig = px.scatter(
                df,
                x=args.x_column,
                y=args.y_columns[0],
                color=args.color_column,
                trendline=trendline_mode,
                title=f"{args.y_columns[0]} vs {args.x_column}",
            )
        else:
            # Multiple y cols — color by series name, ignore explicit color_column
            extra_cols = [args.color_column] if args.color_column and args.color_column not in args.y_columns else []
            plot_cols = [args.x_column, *args.y_columns, *extra_cols]
            melted = df[plot_cols].melt(
                id_vars=[c for c in plot_cols if c != args.x_column and c not in args.y_columns] + [args.x_column],
                value_vars=args.y_columns,
                var_name="series",
                value_name="value",
            )
            fig = px.scatter(
                melted, x=args.x_column, y="value", color="series",
                trendline=trendline_mode,
                title=f"Scatter vs {args.x_column}",
            )
        return fig

    # ── Time Series ─────────────────────────────────────────────────────────
    if plot_type == PlotType.time_series:
        if not args.y_columns:
            raise ValueError("time_series requires at least one y_column")
        _assert_columns_exist(df, [args.x_column, *args.y_columns])
        extra = [args.color_column] if args.color_column and args.color_column not in args.y_columns else []
        plot_df = df[[args.x_column, *args.y_columns, *extra]].copy()
        # Only attempt datetime conversion for string/object columns.
        if pd.api.types.is_string_dtype(plot_df[args.x_column]) or pd.api.types.is_object_dtype(plot_df[args.x_column]):
            plot_df[args.x_column] = pd.to_datetime(plot_df[args.x_column], errors="ignore")
        melted = plot_df.melt(
            id_vars=[args.x_column, *extra],
            value_vars=args.y_columns,
            var_name="series",
            value_name="value",
        )
        fig = px.line(
            melted, x=args.x_column, y="value", color="series",
            title=f"Time series vs {args.x_column}",
        )
        return fig

    # ── Box Plot ────────────────────────────────────────────────────────────
    if plot_type == PlotType.box_plot:
        target_cols = args.y_columns or [args.x_column]
        _assert_columns_exist(df, target_cols)
        if len(target_cols) == 1:
            fig = px.box(
                df, y=target_cols[0],
                x=args.color_column,
                color=args.color_column,
                title=f"Box plot — {target_cols[0]}",
                points="outliers",
            )
        else:
            melted = df[target_cols].melt(var_name="column", value_name="value")
            fig = px.box(
                melted, x="column", y="value", color="column",
                title="Box plots", points="outliers",
            )
        return fig

    # ── Violin Plot ─────────────────────────────────────────────────────────
    if plot_type == PlotType.violin_plot:
        target_cols = args.y_columns or [args.x_column]
        _assert_columns_exist(df, target_cols)
        if len(target_cols) == 1:
            fig = px.violin(
                df, y=target_cols[0],
                x=args.color_column,
                color=args.color_column,
                box=True, points="outliers",
                title=f"Violin plot — {target_cols[0]}",
            )
        else:
            melted = df[target_cols].melt(var_name="column", value_name="value")
            fig = px.violin(
                melted, x="column", y="value", color="column",
                box=True, points="outliers", title="Violin plots",
            )
        return fig

    # ── Pair Plot (Scatter Matrix) ───────────────────────────────────────────
    if plot_type == PlotType.pair_plot:
        cols = args.y_columns or list(df.select_dtypes(include=[np.number]).columns[:6])
        if not cols:
            raise ValueError("pair_plot requires numeric columns in y_columns")
        _assert_columns_exist(df, cols)
        if args.color_column:
            _assert_columns_exist(df, [args.color_column])
        fig = px.scatter_matrix(
            df,
            dimensions=cols,
            color=args.color_column,
            title="Pair plot (scatter matrix)",
        )
        fig.update_traces(diagonal_visible=True, showupperhalf=True)
        return fig

    # ── 3D Scatter ──────────────────────────────────────────────────────────
    if plot_type == PlotType.scatter_3d:
        if not args.y_columns:
            raise ValueError("scatter_3d requires at least one y_column")
        z_col = args.z_column or (args.y_columns[1] if len(args.y_columns) >= 2 else None)
        if not z_col:
            raise ValueError("scatter_3d requires z_column or at least 2 y_columns")
        _assert_columns_exist(df, [args.x_column, args.y_columns[0], z_col])
        if args.color_column:
            _assert_columns_exist(df, [args.color_column])
        fig = px.scatter_3d(
            df,
            x=args.x_column,
            y=args.y_columns[0],
            z=z_col,
            color=args.color_column,
            title=f"3D scatter — {args.x_column} × {args.y_columns[0]} × {z_col}",
            opacity=0.75,
        )
        return fig

    raise ValueError(f"Unsupported plot_type: {plot_type}")



class TransformStrategy(str, Enum):
    log = "log"
    box_cox = "box_cox"
    yeo_johnson = "yeo_johnson"
    sqrt = "sqrt"
    bin_equal_width = "bin_equal_width"
    bin_equal_freq = "bin_equal_freq"
    binarize = "binarize"

class TransformFeaturesArgs(BaseModel):
    dataset_version_id: str = Field(..., min_length=36, max_length=36)
    columns: list[str] = Field(..., min_length=1)
    strategy: TransformStrategy
    kwargs: dict[str, Any] = Field(default_factory=dict, description="e.g. bins for binning or threshold for binarize")

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        return _validate_uuid_str(value)

    @field_validator("columns")
    @classmethod
    def _non_empty_names(cls, value: list[str]) -> list[str]:
        cleaned = [c.strip() for c in value if c and c.strip()]
        if not cleaned:
            raise ValueError("columns must contain at least one non-empty name")
        return cleaned

class EncodeStrategy(str, Enum):
    one_hot = "one_hot"
    ordinal = "ordinal"
    frequency = "frequency"
    target = "target"
    binary = "binary"

class EncodeCategoricalArgs(BaseModel):
    dataset_version_id: str = Field(..., min_length=36, max_length=36)
    columns: list[str] = Field(..., min_length=1)
    strategy: EncodeStrategy
    target_column: Optional[str] = None

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        return _validate_uuid_str(value)

class ReduceDimensionsStrategy(str, Enum):
    pca = "pca"
    variance_threshold = "variance_threshold"

class ReduceDimensionsArgs(BaseModel):
    dataset_version_id: str = Field(..., min_length=36, max_length=36)
    columns: list[str] = Field(..., min_length=1)
    strategy: ReduceDimensionsStrategy
    n_components: Optional[int] = Field(None, description="For PCA")
    threshold: Optional[float] = Field(None, description="For variance threshold")

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        return _validate_uuid_str(value)

class FeatureEngineerStrategy(str, Enum):
    polynomial = "polynomial"
    datetime = "datetime"
    cyclical = "cyclical"
    ratio = "ratio"

class FeatureEngineerArgs(BaseModel):
    dataset_version_id: str = Field(..., min_length=36, max_length=36)
    columns: list[str] = Field(..., min_length=1)
    strategy: FeatureEngineerStrategy
    degree: Optional[int] = Field(2, description="For polynomial")

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        return _validate_uuid_str(value)


async def transform_features(
    dataset_version_id: str,
    columns: list[str],
    strategy: str,
    kwargs: dict[str, Any],
    *,
    session: Optional[AsyncSession] = None,
) -> tuple[str, str]:
    from scipy.stats import boxcox, yeojohnson
    from sklearn.preprocessing import Binarizer, KBinsDiscretizer

    args = TransformFeaturesArgs(
        dataset_version_id=dataset_version_id,
        columns=columns,
        strategy=strategy,  # type: ignore[arg-type]
        kwargs=kwargs,
    )

    async def _run(db: AsyncSession) -> tuple[str, str]:
        parent = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(parent)
        _assert_numeric_columns(df, args.columns)
        result = df.copy()

        for col in args.columns:
            if args.strategy == TransformStrategy.log:
                result[f"{col}_log"] = np.log1p(result[col])
            elif args.strategy == TransformStrategy.sqrt:
                result[f"{col}_sqrt"] = np.sqrt(result[col].clip(lower=0))
            elif args.strategy == TransformStrategy.box_cox:
                val = result[col]
                if (val <= 0).any():
                    val = val - val.min() + 1
                result[f"{col}_boxcox"], _ = boxcox(val)
            elif args.strategy == TransformStrategy.yeo_johnson:
                result[f"{col}_yeo"], _ = yeojohnson(result[col])
            elif args.strategy == TransformStrategy.binarize:
                threshold = args.kwargs.get("threshold", 0.0)
                binarizer = Binarizer(threshold=threshold)
                result[f"{col}_bin"] = binarizer.fit_transform(result[[col]])
            elif args.strategy in (TransformStrategy.bin_equal_width, TransformStrategy.bin_equal_freq):
                strategy_str = "uniform" if args.strategy == TransformStrategy.bin_equal_width else "quantile"
                bins = args.kwargs.get("bins", 5)
                kb = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy=strategy_str) # type: ignore
                result[f"{col}_binned"] = kb.fit_transform(result[[col]])

        summary = f"Applied {args.strategy.value} transformation to {len(args.columns)} column(s)."
        return await _persist_new_version(
            db,
            parent=parent,
            df=result,
            action_performed="transform_features",
            parameters_used={"strategy": args.strategy.value, "columns": args.columns, "kwargs": args.kwargs},
            summary_message=summary,
        )

    if session is not None:
        return await _run(session)
    async with AsyncSessionLocal() as db:
        return await _run(db)


async def encode_categorical(
    dataset_version_id: str,
    columns: list[str],
    strategy: str,
    target_column: Optional[str] = None,
    *,
    session: Optional[AsyncSession] = None,
) -> tuple[str, str]:
    import category_encoders as ce
    args = EncodeCategoricalArgs(
        dataset_version_id=dataset_version_id,
        columns=columns,
        strategy=strategy, # type: ignore[arg-type]
        target_column=target_column,
    )

    async def _run(db: AsyncSession) -> tuple[str, str]:
        parent = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(parent)
        _assert_columns_exist(df, args.columns)
        if args.strategy == EncodeStrategy.target and args.target_column:
            _assert_columns_exist(df, [args.target_column])
        
        result = df.copy()

        if args.strategy == EncodeStrategy.one_hot:
            result = pd.get_dummies(result, columns=args.columns, drop_first=True)
        elif args.strategy == EncodeStrategy.ordinal:
            encoder = ce.OrdinalEncoder(cols=args.columns)
            result = encoder.fit_transform(result)
        elif args.strategy == EncodeStrategy.frequency:
            for col in args.columns:
                freq = result[col].value_counts(normalize=True)
                result[f"{col}_freq"] = result[col].map(freq)
        elif args.strategy == EncodeStrategy.binary:
            encoder = ce.BinaryEncoder(cols=args.columns)
            result = encoder.fit_transform(result)
        elif args.strategy == EncodeStrategy.target and args.target_column:
            encoder = ce.TargetEncoder(cols=args.columns)
            result = encoder.fit_transform(result, result[args.target_column])

        summary = f"Encoded {len(args.columns)} column(s) using {args.strategy.value}."
        return await _persist_new_version(
            db,
            parent=parent,
            df=result,
            action_performed="encode_categorical",
            parameters_used={"strategy": args.strategy.value, "columns": args.columns},
            summary_message=summary,
        )

    if session is not None:
        return await _run(session)
    async with AsyncSessionLocal() as db:
        return await _run(db)


async def reduce_dimensions(
    dataset_version_id: str,
    columns: list[str],
    strategy: str,
    n_components: Optional[int] = None,
    threshold: Optional[float] = None,
    *,
    session: Optional[AsyncSession] = None,
) -> tuple[str, str]:
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import VarianceThreshold

    args = ReduceDimensionsArgs(
        dataset_version_id=dataset_version_id,
        columns=columns,
        strategy=strategy, # type: ignore[arg-type]
        n_components=n_components,
        threshold=threshold,
    )

    async def _run(db: AsyncSession) -> tuple[str, str]:
        parent = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(parent)
        _assert_numeric_columns(df, args.columns)
        result = df.copy()

        if args.strategy == ReduceDimensionsStrategy.pca:
            n_comps = args.n_components or min(len(args.columns), 2)
            pca = PCA(n_components=n_comps)
            comps = pca.fit_transform(result[args.columns].fillna(0))
            for i in range(comps.shape[1]):
                result[f"pca_{i}"] = comps[:, i]
            result = result.drop(columns=args.columns)
        elif args.strategy == ReduceDimensionsStrategy.variance_threshold:
            t = args.threshold or 0.0
            vt = VarianceThreshold(threshold=t)
            vt.fit(result[args.columns].fillna(0))
            to_keep = np.array(args.columns)[vt.get_support()]
            to_drop = [c for c in args.columns if c not in to_keep]
            result = result.drop(columns=to_drop)

        summary = f"Reduced dimensions using {args.strategy.value} on {len(args.columns)} column(s)."
        return await _persist_new_version(
            db,
            parent=parent,
            df=result,
            action_performed="reduce_dimensions",
            parameters_used={"strategy": args.strategy.value},
            summary_message=summary,
        )

    if session is not None:
        return await _run(session)
    async with AsyncSessionLocal() as db:
        return await _run(db)


async def feature_engineer(
    dataset_version_id: str,
    columns: list[str],
    strategy: str,
    degree: Optional[int] = 2,
    *,
    session: Optional[AsyncSession] = None,
) -> tuple[str, str]:
    from sklearn.preprocessing import PolynomialFeatures
    args = FeatureEngineerArgs(
        dataset_version_id=dataset_version_id,
        columns=columns,
        strategy=strategy, # type: ignore[arg-type]
        degree=degree,
    )

    async def _run(db: AsyncSession) -> tuple[str, str]:
        parent = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(parent)
        _assert_columns_exist(df, args.columns)
        result = df.copy()

        if args.strategy == FeatureEngineerStrategy.polynomial:
            _assert_numeric_columns(df, args.columns)
            poly = PolynomialFeatures(degree=args.degree or 2, include_bias=False)
            poly_features = poly.fit_transform(result[args.columns].fillna(0))
            feature_names = poly.get_feature_names_out(args.columns)
            
            for i, name in enumerate(feature_names):
                if name not in result.columns:
                    result[name] = poly_features[:, i]

        elif args.strategy == FeatureEngineerStrategy.datetime:
            for col in args.columns:
                result[col] = pd.to_datetime(result[col], errors='coerce')
                result[f"{col}_year"] = result[col].dt.year
                result[f"{col}_month"] = result[col].dt.month
                result[f"{col}_day"] = result[col].dt.day
                result[f"{col}_hour"] = result[col].dt.hour
                result[f"{col}_dayofweek"] = result[col].dt.dayofweek
        elif args.strategy == FeatureEngineerStrategy.cyclical:
            for col in args.columns:
                max_val = result[col].max()
                if max_val > 0:
                    result[f"{col}_sin"] = np.sin(2 * np.pi * result[col] / max_val)
                    result[f"{col}_cos"] = np.cos(2 * np.pi * result[col] / max_val)
        elif args.strategy == FeatureEngineerStrategy.ratio:
            _assert_numeric_columns(df, args.columns)
            if len(args.columns) == 2:
                result[f"{args.columns[0]}_ratio_{args.columns[1]}"] = result[args.columns[0]] / result[args.columns[1]].replace(0, np.nan)

        summary = f"Engineered features using {args.strategy.value} on {len(args.columns)} column(s)."
        return await _persist_new_version(
            db,
            parent=parent,
            df=result,
            action_performed="feature_engineer",
            parameters_used={"strategy": args.strategy.value},
            summary_message=summary,
        )

    if session is not None:
        return await _run(session)
    async with AsyncSessionLocal() as db:
        return await _run(db)



class BalanceDataStrategy(str, Enum):
    smote = "smote"
    random_undersample = "random_undersample"

class BalanceDataArgs(BaseModel):
    dataset_version_id: str = Field(..., min_length=36, max_length=36)
    target_column: str = Field(..., min_length=1)
    strategy: BalanceDataStrategy
    random_state: int = Field(42, description="Random seed")

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        return _validate_uuid_str(value)


async def balance_data(
    dataset_version_id: str,
    target_column: str,
    strategy: str,
    random_state: int = 42,
    *,
    session: Optional[AsyncSession] = None,
) -> tuple[str, str]:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler

    args = BalanceDataArgs(
        dataset_version_id=dataset_version_id,
        target_column=target_column,
        strategy=strategy, # type: ignore[arg-type]
        random_state=random_state,
    )

    async def _run(db: AsyncSession) -> tuple[str, str]:
        parent = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(parent)
        _assert_columns_exist(df, [args.target_column])
        
        y = df[args.target_column]
        X = df.drop(columns=[args.target_column])

        # Fill missing values with 0 for balancing algorithms
        X_filled = X.fillna(0)

        if args.strategy == BalanceDataStrategy.smote:
            sampler = SMOTE(random_state=args.random_state)
        else:
            sampler = RandomUnderSampler(random_state=args.random_state)
            
        X_res, y_res = sampler.fit_resample(X_filled, y) # type: ignore
        
        result = pd.concat([X_res, y_res], axis=1) # type: ignore

        summary = f"Balanced dataset using {args.strategy.value} on target '{args.target_column}'."
        return await _persist_new_version(
            db,
            parent=parent,
            df=result,
            action_performed="balance_data",
            parameters_used={"strategy": args.strategy.value, "target_column": args.target_column},
            summary_message=summary,
        )

    if session is not None:
        return await _run(session)
    async with AsyncSessionLocal() as db:
        return await _run(db)


# ---------------------------------------------------------------------------
# OpenAI / Groq tool schema export + dispatch map
# ---------------------------------------------------------------------------


LLM_TOOL_SCHEMAS: list[dict[str, Any]] = [
    _openai_tool_schema(
        name="run_duckdb_query",
        description=(
            "Run a read-only SQL aggregation or summary query against a dataset "
            "version loaded into an in-memory DuckDB table named `dataset`. "
            "Returns JSON rows. Does not modify data."
        ),
        args_model=RunDuckDBQueryArgs,
    ),
    _openai_tool_schema(
        name="remove_missing_data",
        description=(
            "Handle missing values using drop_rows, fill_mean, fill_median, or "
            "forward_fill. Saves a new parquet dataset version and logs lineage. "
            "Returns (new_version_id, summary_message)."
        ),
        args_model=RemoveMissingDataArgs,
    ),
    _openai_tool_schema(
        name="remove_outliers",
        description=(
            "Remove outlier rows from selected numeric columns using z_score "
            "(absolute standard deviations) or iqr (IQR fence multiplier). "
            "Saves a new dataset version and logs lineage. "
            "Returns (new_version_id, summary_message)."
        ),
        args_model=RemoveOutliersArgs,
    ),
    _openai_tool_schema(
        name="normalize_data",
        description=(
            "Normalize selected numeric columns using scikit-learn strategies "
            "standard, min_max, or robust. Saves a new dataset version and logs "
            "lineage. Returns (new_version_id, summary_message)."
        ),
        args_model=NormalizeDataArgs,
    ),
    _openai_tool_schema(
        name="generate_custom_plot",
        description=(
            "Build a Plotly figure for a dataset version and return serialized Plotly JSON for React rendering. "
            "Supported plot types: scatter, time_series, histogram, correlation_heatmap, "
            "box_plot (distribution with outliers), violin_plot (distribution with density), "
            "pair_plot (scatter matrix of multiple variables), scatter_3d (3D scatter). "
            "Optional features: color_column (color points by a 3rd variable), "
            "trendline=true (add OLS trendline to scatter), "
            "z_column (Z-axis for scatter_3d), "
            "max_points (auto-subsamples large datasets, default 10000)."
        ),
        args_model=GenerateCustomPlotArgs,
    ),
    _openai_tool_schema(
        name="transform_features",
        description="Apply non-linear transformations (log, box-cox, yeo-johnson, sqrt) or binning.",
        args_model=TransformFeaturesArgs,
    ),
    _openai_tool_schema(
        name="encode_categorical",
        description="Encode categorical features using one_hot, ordinal, frequency, target, or binary.",
        args_model=EncodeCategoricalArgs,
    ),
    _openai_tool_schema(
        name="reduce_dimensions",
        description="Reduce dimensionality via PCA or variance_threshold.",
        args_model=ReduceDimensionsArgs,
    ),
    _openai_tool_schema(
        name="feature_engineer",
        description="Engineer new features (polynomial, datetime extraction, cyclical, ratio).",
        args_model=FeatureEngineerArgs,
    ),
    _openai_tool_schema(
        name="balance_data",
        description="Balance dataset using smote or random_undersample.",
        args_model=BalanceDataArgs,
    ),
    _openai_tool_schema(
        name="drop_columns",
        description=(
            "Drop the specified columns from the dataset. "
            "Saves a new dataset version and logs lineage. "
            "Returns (new_version_id, summary_message)."
        ),
        args_model=DropColumnsArgs,
    ),
    _openai_tool_schema(
        name="execute_formula",
        description=(
            "Create a new column by evaluating a mathematical formula over existing columns "
            "(e.g., 'colA * colB' or 'colA ** 2'). "
            "Saves a new dataset version and logs lineage. "
            "Returns (new_version_id, summary_message)."
        ),
        args_model=ExecuteFormulaArgs,
    ),
    _openai_tool_schema(
        name="rename_columns",
        description=(
            "Rename columns in the dataset using a mapping of old names to new names. "
            "Saves a new dataset version and logs lineage. "
            "Returns (new_version_id, summary_message)."
        ),
        args_model=RenameColumnsArgs,
    ),
    # Time-series tools
    _openai_tool_schema(
        name="rolling_aggregate",
        description=(
            "Compute rolling window statistics (mean, std, min, max, sum) over time-series columns. "
            "Adds new columns named '{col}_rolling{window}_{agg_func}'. "
            "Saves a new dataset version and logs lineage."
        ),
        args_model=RollingAggregateArgs,
    ),
    _openai_tool_schema(
        name="create_lag_features",
        description=(
            "Create lagged (time-shifted) versions of columns for time-series modeling. "
            "For each column and each lag period n, adds a column '{col}_lag{n}'. "
            "Saves a new dataset version and logs lineage."
        ),
        args_model=CreateLagFeaturesArgs,
    ),
    # Advanced feature selection & data quality tools
    _openai_tool_schema(
        name="select_features_rfe",
        description=(
            "Use Recursive Feature Elimination (RFE) with Ridge or Lasso to automatically "
            "select the most important features and drop the rest. "
            "Saves a new dataset version and logs lineage."
        ),
        args_model=SelectFeaturesRFEArgs,
    ),
    _openai_tool_schema(
        name="detect_anomalies",
        description=(
            "Detect multivariate anomalies using Isolation Forest. "
            "Can either drop anomalous rows or add an 'is_anomaly' flag column. "
            "Saves a new dataset version and logs lineage."
        ),
        args_model=DetectAnomaliesArgs,
    ),
    _openai_tool_schema(
        name="drop_collinear_features",
        description=(
            "Drop highly-correlated (collinear) feature columns to improve model stability. "
            "For each pair of columns with |corr| > threshold, drops the one with lower variance. "
            "Saves a new dataset version and logs lineage."
        ),
        args_model=DropCollinearFeaturesArgs,
    ),
    # Statistical profiler
    _openai_tool_schema(
        name="get_column_statistics",
        description=(
            "Compute per-column descriptive statistics: count, null count, null%, mean, std, "
            "min, 25th/50th/75th percentiles, and max for numeric columns; "
            "dtype and top values for categorical columns. "
            "Does not modify data. Use this before any analysis or when the user asks about data quality."
        ),
        args_model=GetColumnStatisticsArgs,
    ),
]

TOOL_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "run_duckdb_query": run_duckdb_query,
    "remove_missing_data": remove_missing_data,
    "remove_outliers": remove_outliers,
    "normalize_data": normalize_data,
    "generate_custom_plot": generate_custom_plot,
    "transform_features": transform_features,
    "encode_categorical": encode_categorical,
    "reduce_dimensions": reduce_dimensions,
    "feature_engineer": feature_engineer,
    "balance_data": balance_data,
    "drop_columns": drop_columns,
    "execute_formula": execute_formula,
    "rename_columns": rename_columns,
    # Time-series
    "rolling_aggregate": rolling_aggregate,
    "create_lag_features": create_lag_features,
    # Advanced feature selection & data quality
    "select_features_rfe": select_features_rfe,
    "detect_anomalies": detect_anomalies,
    "drop_collinear_features": drop_collinear_features,
    # Statistical profiler
    "get_column_statistics": get_column_statistics,
}

__all__ = [
    "GenerateCustomPlotArgs",
    "LLM_TOOL_SCHEMAS",
    "MissingStrategy",
    "NormalizeDataArgs",
    "NormalizeStrategy",
    "OutlierMethod",
    "PlotType",
    "RemoveMissingDataArgs",
    "RemoveOutliersArgs",
    "RunDuckDBQueryArgs",
    "DropColumnsArgs",
    "ExecuteFormulaArgs",
    "RenameColumnsArgs",
    "RollingAggregateArgs",
    "CreateLagFeaturesArgs",
    "SelectFeaturesRFEArgs",
    "DetectAnomaliesArgs",
    "DropCollinearFeaturesArgs",
    "GetColumnStatisticsArgs",
    "TOOL_FUNCTIONS",
    "generate_custom_plot",
    "normalize_data",
    "remove_missing_data",
    "remove_outliers",
    "drop_columns",
    "execute_formula",
    "rename_columns",
    "rolling_aggregate",
    "create_lag_features",
    "select_features_rfe",
    "detect_anomalies",
    "drop_collinear_features",
    "get_column_statistics",
    "run_duckdb_query",
]
