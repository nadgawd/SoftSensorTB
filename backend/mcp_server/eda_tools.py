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
from pydantic import BaseModel, Field, field_validator, model_validator
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
    clip = "clip"


class OutlierAction(str, Enum):
    drop = "drop"
    cap = "cap"


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
    columns: Optional[list[str]] = Field(
        default=None,
        description=(
            "Columns to clean. Omit to clean every column. drop_rows drops rows with a "
            "missing value in any of these columns; mean/median/knn/iterative use the numeric ones."
        ),
    )
    flag_missing: bool = Field(
        default=False,
        description="If True, creates boolean indicator columns (e.g. col_is_missing) before imputing.",
    )
    n_neighbors: int = Field(
        default=5,
        description="Number of neighbours used by the knn strategy.",
        ge=1,
    )
    max_iter: int = Field(
        default=10,
        description="Maximum imputation rounds for the iterative strategy.",
        ge=1,
        le=100,
    )
    random_state: int = Field(
        default=42,
        description="Random seed for the iterative strategy.",
    )

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        return _validate_uuid_str(value)

    @field_validator("columns")
    @classmethod
    def _optional_names(cls, value: Optional[list[str]]) -> Optional[list[str]]:
        return _clean_optional_names(value)


_DEFAULT_OUTLIER_THRESHOLDS = {
    OutlierMethod.z_score: 3.0,
    OutlierMethod.iqr: 1.5,
    OutlierMethod.winsorize: 0.05,
}


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
        description=(
            "z_score and iqr find outliers beyond a z-score or IQR fence, then drop the rows "
            "or cap the values at the fence (see action); winsorize caps the lowest and highest "
            "threshold fraction at those percentiles; clip caps values to lower_bound / upper_bound."
        ),
    )
    threshold: Optional[float] = Field(
        default=None,
        description=(
            "For z_score: absolute z-score cutoff (default 3). "
            "For iqr: multiplier on the IQR fence (default 1.5). "
            "For winsorize: fraction capped in each tail, below 0.5 (default 0.05)."
        ),
        gt=0,
    )
    action: OutlierAction = Field(
        default=OutlierAction.drop,
        description=(
            "For z_score and iqr: drop the outlier rows, or cap (clip) the outlying values "
            "at the fence and keep every row."
        ),
    )
    lower_bound: Optional[float] = Field(
        default=None,
        description="For clip: values below this are raised to it. Omit for no lower limit.",
    )
    upper_bound: Optional[float] = Field(
        default=None,
        description="For clip: values above this are lowered to it. Omit for no upper limit.",
    )

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        return _validate_uuid_str(value)

    @field_validator("columns")
    @classmethod
    def _non_empty_names(cls, value: list[str]) -> list[str]:
        return _clean_names(value)

    @model_validator(mode="after")
    def _method_limits(self) -> "RemoveOutliersArgs":
        if self.method == OutlierMethod.winsorize and self.threshold is not None and self.threshold >= 0.5:
            raise ValueError("For winsorize, threshold is the fraction capped in each tail and must be below 0.5")
        if self.method == OutlierMethod.clip:
            if self.lower_bound is None and self.upper_bound is None:
                raise ValueError("clip needs lower_bound, upper_bound or both")
            if self.lower_bound is not None and self.upper_bound is not None and self.lower_bound >= self.upper_bound:
                raise ValueError("lower_bound must be smaller than upper_bound")
        return self


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
        description="Scaling strategy: standard, min_max, robust, or maxabs.",
    )
    feature_min: float = Field(
        default=0.0,
        description="Lower end of the target range for min_max scaling.",
    )
    feature_max: float = Field(
        default=1.0,
        description="Upper end of the target range for min_max scaling.",
    )
    quantile_low: float = Field(
        default=25.0,
        description="Lower percentile of the range used by robust scaling.",
        ge=0,
        le=100,
    )
    quantile_high: float = Field(
        default=75.0,
        description="Upper percentile of the range used by robust scaling.",
        ge=0,
        le=100,
    )

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        return _validate_uuid_str(value)

    @field_validator("columns")
    @classmethod
    def _non_empty_names(cls, value: list[str]) -> list[str]:
        return _clean_names(value)

    @model_validator(mode="after")
    def _valid_ranges(self) -> "NormalizeDataArgs":
        if self.feature_min >= self.feature_max:
            raise ValueError("feature_min must be smaller than feature_max")
        if self.quantile_low >= self.quantile_high:
            raise ValueError("quantile_low must be smaller than quantile_high")
        return self


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

class RollingAggFunc(str, Enum):
    mean = "mean"
    std = "std"
    min = "min"
    max = "max"
    sum = "sum"
    median = "median"


class RollingAggregateArgs(BaseModel):
    """Arguments for computing rolling window aggregates."""

    dataset_version_id: str = Field(..., description="UUID of the dataset version.", min_length=36, max_length=36)
    columns: list[str] = Field(..., description="Numeric columns to compute rolling aggregates for.", min_length=1)
    window: int = Field(..., description="Rolling window size (number of rows).", ge=2)
    agg_func: RollingAggFunc = Field(RollingAggFunc.mean, description="Aggregation function: mean, std, min, max, sum, or median.")
    min_periods: int = Field(1, description="Minimum number of observations required to produce a value.", ge=1)
    center: bool = Field(False, description="If True, centre the window on each row instead of looking back only.")

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, v: str) -> str: return _validate_uuid_str(v)

    @field_validator("columns")
    @classmethod
    def _non_empty_names(cls, value: list[str]) -> list[str]:
        return _clean_names(value)

    @model_validator(mode="after")
    def _periods_fit_window(self) -> "RollingAggregateArgs":
        if self.min_periods > self.window:
            raise ValueError("min_periods cannot be larger than window")
        return self


class CreateLagFeaturesArgs(BaseModel):
    """Arguments for creating lag/shift features."""

    dataset_version_id: str = Field(..., description="UUID of the dataset version.", min_length=36, max_length=36)
    columns: list[str] = Field(..., description="Columns to create lag features for.", min_length=1)
    lag_periods: list[int] = Field(..., description="List of lag periods (positive integers, e.g. [1, 5, 10]).", min_length=1)

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, v: str) -> str: return _validate_uuid_str(v)

    @field_validator("columns")
    @classmethod
    def _non_empty_names(cls, value: list[str]) -> list[str]:
        return _clean_names(value)

    @field_validator("lag_periods")
    @classmethod
    def _valid_lags(cls, v: list[int]) -> list[int]:
        if any(p <= 0 for p in v):
            raise ValueError("All lag_periods must be positive integers")
        return sorted(set(v))


# ---------------------------------------------------------------------------
# Advanced feature selection & data quality tools
# ---------------------------------------------------------------------------

class SelectFeaturesRFEArgs(BaseModel):
    """Arguments for Recursive Feature Elimination."""

    dataset_version_id: str = Field(..., description="UUID of the dataset version.", min_length=36, max_length=36)
    target_column: str = Field(..., description="Name of the target/response column.")
    feature_columns: Optional[list[str]] = Field(
        None,
        description="Candidate feature columns. Omit to consider every numeric column except the target.",
    )
    n_features_to_select: int = Field(5, description="Number of top features to keep.", ge=1)
    estimator: str = Field(
        "ridge",
        description="Base estimator for RFE: 'linear' (plain linear regression), 'ridge' or 'lasso'.",
        json_schema_extra={"enum": ["linear", "ridge", "lasso"]},
    )
    alpha: float = Field(1.0, description="Regularization strength of the ridge or lasso estimator.", gt=0)
    step: int = Field(1, description="Number of features removed at each elimination round.", ge=1)

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, v: str) -> str: return _validate_uuid_str(v)

    @field_validator("feature_columns")
    @classmethod
    def _optional_names(cls, value: Optional[list[str]]) -> Optional[list[str]]:
        return _clean_optional_names(value)

    @field_validator("estimator")
    @classmethod
    def _valid_estimator(cls, v: str) -> str:
        v = v.strip().lower()
        if v in {"ols", "linear_regression", "linear regression"}:
            v = "linear"
        if v not in {"linear", "ridge", "lasso"}:
            raise ValueError("estimator must be 'linear', 'ridge' or 'lasso'")
        return v


class DetectAnomaliesArgs(BaseModel):
    """Arguments for Isolation Forest anomaly detection."""

    dataset_version_id: str = Field(..., description="UUID of the dataset version.", min_length=36, max_length=36)
    columns: Optional[list[str]] = Field(
        None,
        description="Numeric columns the Isolation Forest looks at. Omit to use every numeric column.",
    )
    contamination: float = Field(0.05, description="Expected proportion of anomalies in the dataset (0.01–0.5).", ge=0.01, le=0.5)
    drop_anomalies: bool = Field(True, description="If True, drop anomalous rows. If False, add an 'is_anomaly' flag column instead.")
    n_estimators: int = Field(100, description="Number of trees in the Isolation Forest.", ge=10, le=2000)
    random_state: int = Field(42, description="Random seed of the Isolation Forest.")

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, v: str) -> str: return _validate_uuid_str(v)

    @field_validator("columns")
    @classmethod
    def _optional_names(cls, value: Optional[list[str]]) -> Optional[list[str]]:
        return _clean_optional_names(value)


class CorrelationMethod(str, Enum):
    pearson = "pearson"
    spearman = "spearman"
    kendall = "kendall"


class DropCollinearFeaturesArgs(BaseModel):
    """Arguments for dropping highly-correlated (collinear) features."""

    dataset_version_id: str = Field(..., description="UUID of the dataset version.", min_length=36, max_length=36)
    threshold: float = Field(0.95, description="Absolute correlation above which one column of a pair is dropped.", ge=0.5, le=1.0)
    target_column: Optional[str] = Field(None, description="If provided, this column is excluded from the collinearity analysis and never dropped.")
    method: CorrelationMethod = Field(CorrelationMethod.pearson, description="Correlation coefficient: pearson, spearman, or kendall.")

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
    x_column: Optional[str] = Field(
        None,
        description=(
            "X-axis / primary column. Required for scatter, time_series and scatter_3d; "
            "ignored for correlation_heatmap and pair_plot."
        ),
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

    @field_validator("x_column", "color_column", "z_column")
    @classmethod
    def _blank_is_none(cls, value: Optional[str]) -> Optional[str]:
        return value.strip() or None if isinstance(value, str) else value

    @field_validator("y_columns")
    @classmethod
    def _clean_y(cls, value: list[str]) -> list[str]:
        return [c.strip() for c in value if c and c.strip()]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _clean_names(value: list[str]) -> list[str]:
    cleaned = [c.strip() for c in value if c and c.strip()]
    if not cleaned:
        raise ValueError("columns must contain at least one non-empty name")
    return cleaned


def _clean_optional_names(value: Optional[list[str]]) -> Optional[list[str]]:
    """An omitted or empty list means "all applicable columns"."""
    if value is None:
        return None
    return [c.strip() for c in value if c and c.strip()] or None

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
            new_col_series = df.eval(args.formula)
        except Exception as e:
            raise ValueError(f"Failed to evaluate formula {args.formula!r}: {e}") from e
        df[args.new_column_name] = new_col_series
        
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
    center: bool = False,
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
        agg_func=agg_func,  # type: ignore[arg-type]
        min_periods=min_periods,
        center=center,
    )
    func = args.agg_func.value

    async def _run(db: AsyncSession) -> tuple[str, str]:
        parent = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(parent)
        _assert_numeric_columns(df, args.columns)

        result = df.copy()
        created: list[str] = []
        for col in args.columns:
            new_col_name = f"{col}_rolling{args.window}_{func}"
            roller = result[col].rolling(window=args.window, min_periods=args.min_periods, center=args.center)
            result[new_col_name] = getattr(roller, func)()
            created.append(new_col_name)

        return await _persist_new_version(
            db,
            parent=parent,
            df=result,
            action_performed="rolling_aggregate",
            parameters_used={
                "columns": args.columns,
                "window": args.window,
                "agg_func": func,
                "min_periods": args.min_periods,
                "center": args.center,
            },
            summary_message=f"Created {len(created)} rolling {func}(window={args.window}) column(s): {', '.join(created)}",
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
    feature_columns: Optional[list[str]] = None,
    alpha: float = 1.0,
    step: int = 1,
    *,
    session: Optional[AsyncSession] = None,
) -> tuple[str, str]:
    """Drop low-importance features using Recursive Feature Elimination (RFE).

    Returns:
        (new_version_id, summary_message)
    """
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LinearRegression, Ridge, Lasso

    args = SelectFeaturesRFEArgs(
        dataset_version_id=dataset_version_id,
        target_column=target_column,
        feature_columns=feature_columns,
        n_features_to_select=n_features_to_select,
        estimator=estimator,
        alpha=alpha,
        step=step,
    )

    async def _run(db: AsyncSession) -> tuple[str, str]:
        parent = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(parent)
        _assert_numeric_columns(df, [args.target_column])

        if args.feature_columns:
            feature_cols = [c for c in args.feature_columns if c != args.target_column]
            _assert_numeric_columns(df, feature_cols)
        else:
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

        base_est = {
            "linear": LinearRegression,
            "ridge": lambda: Ridge(alpha=args.alpha),
            "lasso": lambda: Lasso(alpha=args.alpha),
        }[args.estimator]()
        rfe = RFE(estimator=base_est, n_features_to_select=args.n_features_to_select, step=args.step)
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
                "alpha": args.alpha,
                "step": args.step,
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
    columns: Optional[list[str]] = None,
    n_estimators: int = 100,
    random_state: int = 42,
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
        columns=columns,
        contamination=contamination,
        drop_anomalies=drop_anomalies,
        n_estimators=n_estimators,
        random_state=random_state,
    )

    async def _run(db: AsyncSession) -> tuple[str, str]:
        parent = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(parent)

        if args.columns:
            _assert_numeric_columns(df, args.columns)
            numeric_cols = list(args.columns)
        else:
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            return str(parent.id), "No numeric columns found for anomaly detection."

        X = df[numeric_cols].fillna(0)
        iso = IsolationForest(
            n_estimators=args.n_estimators,
            contamination=args.contamination,
            random_state=args.random_state,
            n_jobs=-1,
        )
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
                "columns": numeric_cols,
                "contamination": args.contamination,
                "drop_anomalies": args.drop_anomalies,
                "n_estimators": args.n_estimators,
                "random_state": args.random_state,
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
    method: str = "pearson",
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
        method=method,  # type: ignore[arg-type]
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

        corr_matrix = df[feature_cols].corr(method=args.method.value).abs()
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
            parameters_used={
                "threshold": args.threshold,
                "method": args.method.value,
                "dropped_columns": sorted(to_drop),
            },
            summary_message=(
                f"Dropped {len(to_drop)} collinear feature(s) with |{args.method.value} corr| > {args.threshold}: "
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
    columns: Optional[list[str]] = None,
    n_neighbors: int = 5,
    max_iter: int = 10,
    random_state: int = 42,
    *,
    session: Optional[AsyncSession] = None,
) -> tuple[str, str]:
    """Remove or impute missing values; save a new version and log lineage."""
    args = RemoveMissingDataArgs(
        dataset_version_id=dataset_version_id,
        strategy=strategy,  # type: ignore[arg-type]
        columns=columns,
        flag_missing=flag_missing,
        n_neighbors=n_neighbors,
        max_iter=max_iter,
        random_state=random_state,
    )

    async def _run(db: AsyncSession) -> tuple[str, str]:
        parent = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(parent)
        initial_rows = len(df)
        if args.columns:
            _assert_columns_exist(df, args.columns)
        target_cols = list(args.columns or df.columns)

        # Coerce CSV "fake" nulls (empty strings, whitespace, "NA", "N/A",
        # "null", "NaN", "nan", "None", "?") into real np.nan so that
        # dropna() and fillna() strategies can detect them.
        df = _coerce_fake_nulls(df)

        before_na = int(df[target_cols].isna().sum().sum())

        if args.flag_missing:
            for col in target_cols:
                if df[col].isna().any():
                    df[f"{col}_is_missing"] = df[col].isna().astype(int)

        numeric_cols = [c for c in target_cols if pd.api.types.is_numeric_dtype(df[c])]
        result = df.copy()
        if args.strategy == MissingStrategy.drop_rows:
            result = df.dropna(subset=target_cols)
        elif args.strategy == MissingStrategy.fill_mean:
            result[numeric_cols] = result[numeric_cols].fillna(result[numeric_cols].mean())
        elif args.strategy == MissingStrategy.fill_median:
            result[numeric_cols] = result[numeric_cols].fillna(result[numeric_cols].median())
        elif args.strategy == MissingStrategy.fill_mode:
            for col in target_cols:
                mode_vals = result[col].mode()
                if not mode_vals.empty:
                    result[col] = result[col].fillna(mode_vals[0])
        elif args.strategy == MissingStrategy.forward_fill:
            result[target_cols] = result[target_cols].ffill()
        elif args.strategy == MissingStrategy.backward_fill:
            result[target_cols] = result[target_cols].bfill()
        elif args.strategy in (MissingStrategy.knn, MissingStrategy.iterative) and numeric_cols:
            if args.strategy == MissingStrategy.knn:
                imputer = KNNImputer(n_neighbors=args.n_neighbors)
            else:
                imputer = IterativeImputer(max_iter=args.max_iter, random_state=args.random_state)
            result[numeric_cols] = imputer.fit_transform(result[numeric_cols])

        final_rows = len(result)
        rows_dropped = initial_rows - final_rows
        after_na = int(result[target_cols].isna().sum().sum())
        scope = f" on {len(target_cols)} selected column(s)" if args.columns else ""
        summary = (
            f"Applied missing-data strategy '{args.strategy.value}'{scope}: "
            f"rows {initial_rows} → {final_rows} "
            f"({rows_dropped} row(s) dropped), "
            f"null cells {before_na} → {after_na}."
        )
        parameters: dict[str, Any] = {
            "strategy": args.strategy.value,
            "flag_missing": args.flag_missing,
            "columns": args.columns,
        }
        if args.strategy == MissingStrategy.knn:
            parameters["n_neighbors"] = args.n_neighbors
        if args.strategy == MissingStrategy.iterative:
            parameters.update(max_iter=args.max_iter, random_state=args.random_state)
        return await _persist_new_version(
            db,
            parent=parent,
            df=result,
            action_performed="remove_missing_data",
            parameters_used=parameters,
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
    threshold: Optional[float] = None,
    action: str = "drop",
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
    *,
    session: Optional[AsyncSession] = None,
) -> tuple[str, str]:
    """Drop or cap outliers (z-score / IQR fences, winsorizing, fixed bounds); save version + lineage."""
    args = RemoveOutliersArgs(
        dataset_version_id=dataset_version_id,
        columns=columns,
        method=method,  # type: ignore[arg-type]
        threshold=threshold,
        action=action,  # type: ignore[arg-type]
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    if args.threshold is None:
        args.threshold = _DEFAULT_OUTLIER_THRESHOLDS.get(args.method)

    async def _run(db: AsyncSession) -> tuple[str, str]:
        parent = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(parent)
        _assert_numeric_columns(df, args.columns)

        result = df.copy()
        parameters: dict[str, Any] = {"columns": args.columns, "method": args.method.value}
        if args.method == OutlierMethod.winsorize:
            # Caps the lowest and highest `threshold` fraction instead of dropping rows.
            limit = float(args.threshold)
            for col in args.columns:
                result[col] = mstats.winsorize(result[col], limits=[limit, limit])
            parameters["threshold"] = limit
            summary = (
                f"Winsorized columns {args.columns} using limits "
                f"[{limit}, {1 - limit}]."
            )
        elif args.method == OutlierMethod.clip:
            capped = 0
            for col in args.columns:
                series = df[col]
                too_low = series < args.lower_bound if args.lower_bound is not None else False
                too_high = series > args.upper_bound if args.upper_bound is not None else False
                capped += int((too_low | too_high).sum())
                result[col] = series.clip(lower=args.lower_bound, upper=args.upper_bound)
            parameters.update(lower_bound=args.lower_bound, upper_bound=args.upper_bound)
            summary = (
                f"Clipped {capped} value(s) in columns {args.columns} to "
                f"[{'-∞' if args.lower_bound is None else args.lower_bound}, "
                f"{'∞' if args.upper_bound is None else args.upper_bound}]; no rows dropped."
            )
        else:
            fences: dict[str, tuple[float, float]] = {}
            for col in args.columns:
                series = df[col]
                if args.method == OutlierMethod.z_score:
                    std = series.std(ddof=0)
                    if std == 0 or pd.isna(std):
                        continue
                    mean = series.mean()
                    fences[col] = (mean - args.threshold * std, mean + args.threshold * std)
                else:  # iqr
                    q1 = series.quantile(0.25)
                    q3 = series.quantile(0.75)
                    iqr = q3 - q1
                    fences[col] = (q1 - args.threshold * iqr, q3 + args.threshold * iqr)
            parameters.update(threshold=args.threshold, action=args.action.value)

            if args.action == OutlierAction.cap:
                capped = 0
                for col, (lower, upper) in fences.items():
                    capped += int(((df[col] < lower) | (df[col] > upper)).sum())
                    result[col] = df[col].clip(lower=lower, upper=upper)
                summary = (
                    f"Capped {capped} outlier value(s) at the {args.method.value} fence "
                    f"(threshold={args.threshold}) in columns {args.columns}; no rows dropped."
                )
            else:
                mask = pd.Series(True, index=df.index)
                for col, (lower, upper) in fences.items():
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
            parameters_used=parameters,
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
    feature_min: float = 0.0,
    feature_max: float = 1.0,
    quantile_low: float = 25.0,
    quantile_high: float = 75.0,
    *,
    session: Optional[AsyncSession] = None,
) -> tuple[str, str]:
    """Scale columns with scikit-learn; save a new version and log lineage."""
    args = NormalizeDataArgs(
        dataset_version_id=dataset_version_id,
        columns=columns,
        strategy=strategy,  # type: ignore[arg-type]
        feature_min=feature_min,
        feature_max=feature_max,
        quantile_low=quantile_low,
        quantile_high=quantile_high,
    )

    parameters: dict[str, Any] = {"columns": args.columns, "strategy": args.strategy.value}
    if args.strategy == NormalizeStrategy.min_max:
        scaler: Any = MinMaxScaler(feature_range=(args.feature_min, args.feature_max))
        parameters["feature_range"] = [args.feature_min, args.feature_max]
    elif args.strategy == NormalizeStrategy.robust:
        scaler = RobustScaler(quantile_range=(args.quantile_low, args.quantile_high))
        parameters["quantile_range"] = [args.quantile_low, args.quantile_high]
    elif args.strategy == NormalizeStrategy.maxabs:
        scaler = MaxAbsScaler()
    else:
        scaler = StandardScaler()

    async def _run(db: AsyncSession) -> tuple[str, str]:
        parent = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(parent)
        _assert_numeric_columns(df, args.columns)

        result = df.copy()
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
            parameters_used=parameters,
            summary_message=summary,
        )

    if session is not None:
        return await _run(session)
    async with AsyncSessionLocal() as db:
        return await _run(db)


async def generate_custom_plot(
    dataset_version_id: str,
    plot_type: str,
    x_column: Optional[str] = None,
    y_columns: Optional[list[str]] = None,
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
        y_columns=y_columns or [],
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
    if plot_type in (PlotType.scatter, PlotType.time_series, PlotType.scatter_3d) and not args.x_column:
        raise ValueError(f"{plot_type.value} needs x_column")
    if plot_type in (PlotType.histogram, PlotType.box_plot, PlotType.violin_plot) and not (args.x_column or args.y_columns):
        raise ValueError(f"{plot_type.value} needs x_column or y_columns")

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
            parsed = pd.to_datetime(plot_df[args.x_column], errors="coerce", format="mixed")
            # Text that is not dates (batch ids, tags) stays as it is.
            if parsed.notna().sum() == plot_df[args.x_column].notna().sum():
                plot_df[args.x_column] = parsed
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
    columns: list[str] = Field(..., min_length=1, description="Numeric columns to transform; results are added as new columns.")
    strategy: TransformStrategy = Field(
        ...,
        description="log, box_cox, yeo_johnson, sqrt, bin_equal_width, bin_equal_freq, or binarize.",
    )
    n_bins: int = Field(5, description="Number of bins for bin_equal_width and bin_equal_freq.", ge=2, le=100)
    threshold: float = Field(0.0, description="Values above this become 1 for binarize, the rest 0.")

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
    label = "label"
    ordinal = "ordinal"
    frequency = "frequency"
    target = "target"
    binary = "binary"

class EncodeCategoricalArgs(BaseModel):
    dataset_version_id: str = Field(..., min_length=36, max_length=36)
    columns: list[str] = Field(..., min_length=1, description="Categorical columns to encode.")
    strategy: EncodeStrategy = Field(
        ...,
        description=(
            "one_hot (dummy columns), label (codes 0..n-1 in sorted order, like scikit-learn's "
            "LabelEncoder), ordinal (codes 1..n in sorted order or in category_order), frequency, "
            "target (needs target_column), or binary."
        ),
    )
    target_column: Optional[str] = Field(None, description="Numeric target column; required by the target strategy.")
    drop_first: bool = Field(True, description="For one_hot: drop the first category of each column to avoid redundant dummies.")
    smoothing: float = Field(
        1.0,
        description="For target: pulls the mean of rare categories towards the overall mean (0 = plain category mean).",
        ge=0,
    )
    category_order: Optional[list[str]] = Field(
        None,
        description=(
            "For ordinal: the categories from lowest to highest, e.g. ['low', 'medium', 'high'], "
            "coded 1..n in this order. Every value of the columns must be listed. Omit to use sorted order."
        ),
    )

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        return _validate_uuid_str(value)

    @field_validator("columns")
    @classmethod
    def _non_empty_names(cls, value: list[str]) -> list[str]:
        return _clean_names(value)

    @field_validator("category_order")
    @classmethod
    def _clean_order(cls, value: Optional[list[str]]) -> Optional[list[str]]:
        if value is None:
            return None
        cleaned = [str(c).strip() for c in value if str(c).strip()]
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("category_order lists a category more than once")
        return cleaned or None

    @model_validator(mode="after")
    def _target_needed(self) -> "EncodeCategoricalArgs":
        if self.strategy == EncodeStrategy.target:
            if not (self.target_column or "").strip():
                raise ValueError("The target strategy needs target_column")
            if self.target_column in self.columns:
                raise ValueError("target_column cannot also be one of the columns to encode")
        return self

class ReduceDimensionsStrategy(str, Enum):
    pca = "pca"
    variance_threshold = "variance_threshold"

class ReduceDimensionsArgs(BaseModel):
    dataset_version_id: str = Field(..., min_length=36, max_length=36)
    columns: list[str] = Field(..., min_length=1, description="Numeric columns to reduce.")
    strategy: ReduceDimensionsStrategy = Field(..., description="pca or variance_threshold.")
    n_components: int = Field(2, description="Number of principal components to keep (pca).", ge=1)
    keep_original: bool = Field(False, description="For pca: keep the original columns next to the components.")
    threshold: float = Field(0.0, description="Drop columns whose variance is not above this (variance_threshold).", ge=0)

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        return _validate_uuid_str(value)

    @field_validator("columns")
    @classmethod
    def _non_empty_names(cls, value: list[str]) -> list[str]:
        return _clean_names(value)

class FeatureEngineerStrategy(str, Enum):
    polynomial = "polynomial"
    datetime = "datetime"
    cyclical = "cyclical"
    ratio = "ratio"

class FeatureEngineerArgs(BaseModel):
    dataset_version_id: str = Field(..., min_length=36, max_length=36)
    columns: list[str] = Field(..., min_length=1, description="Columns to build features from (exactly 2 for ratio).")
    strategy: FeatureEngineerStrategy = Field(
        ...,
        description="polynomial, datetime (year/month/day/hour/weekday parts), cyclical (sin/cos), or ratio (first / second column).",
    )
    degree: int = Field(2, description="Polynomial degree.", ge=2, le=5)
    interaction_only: bool = Field(False, description="For polynomial: only products of different columns, no powers.")
    period: Optional[float] = Field(
        None,
        description="For cyclical: length of one cycle (e.g. 24 for hours). Omit to use each column's maximum.",
        gt=0,
    )

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        return _validate_uuid_str(value)

    @field_validator("columns")
    @classmethod
    def _non_empty_names(cls, value: list[str]) -> list[str]:
        return _clean_names(value)

    @model_validator(mode="after")
    def _ratio_pair(self) -> "FeatureEngineerArgs":
        if self.strategy == FeatureEngineerStrategy.ratio and len(self.columns) != 2:
            raise ValueError("The ratio strategy needs exactly 2 columns (numerator, denominator)")
        return self


async def transform_features(
    dataset_version_id: str,
    columns: list[str],
    strategy: str,
    n_bins: int = 5,
    threshold: float = 0.0,
    *,
    session: Optional[AsyncSession] = None,
) -> tuple[str, str]:
    from scipy.stats import boxcox, yeojohnson
    from sklearn.preprocessing import Binarizer, KBinsDiscretizer

    args = TransformFeaturesArgs(
        dataset_version_id=dataset_version_id,
        columns=columns,
        strategy=strategy,  # type: ignore[arg-type]
        n_bins=n_bins,
        threshold=threshold,
    )
    parameters: dict[str, Any] = {"strategy": args.strategy.value, "columns": args.columns}
    if args.strategy in (TransformStrategy.bin_equal_width, TransformStrategy.bin_equal_freq):
        parameters["n_bins"] = args.n_bins
    elif args.strategy == TransformStrategy.binarize:
        parameters["threshold"] = args.threshold

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
                binarizer = Binarizer(threshold=args.threshold)
                result[f"{col}_bin"] = binarizer.fit_transform(result[[col]])
            elif args.strategy in (TransformStrategy.bin_equal_width, TransformStrategy.bin_equal_freq):
                strategy_str = "uniform" if args.strategy == TransformStrategy.bin_equal_width else "quantile"
                kb = KBinsDiscretizer(n_bins=args.n_bins, encode='ordinal', strategy=strategy_str) # type: ignore
                result[f"{col}_binned"] = kb.fit_transform(result[[col]])

        summary = f"Applied {args.strategy.value} transformation to {len(args.columns)} column(s)."
        return await _persist_new_version(
            db,
            parent=parent,
            df=result,
            action_performed="transform_features",
            parameters_used=parameters,
            summary_message=summary,
        )

    if session is not None:
        return await _run(session)
    async with AsyncSessionLocal() as db:
        return await _run(db)


def _ordinal_codes(series: pd.Series) -> pd.Series:
    """Categories numbered 1..n in sorted order; missing values stay missing."""
    codes, _ = pd.factorize(series.astype("string"), sort=True)
    return pd.Series(np.where(codes >= 0, codes + 1, np.nan), index=series.index)


def _label_codes(series: pd.Series) -> pd.Series:
    """Categories numbered 0..n-1 in sorted order (scikit-learn LabelEncoder); missing stays missing."""
    return _ordinal_codes(series) - 1


def _ordered_codes(series: pd.Series, order: Sequence[str]) -> pd.Series:
    """Categories numbered 1..n in the given order; every value must be listed."""
    text = series.astype("string")
    mapping = {category: i for i, category in enumerate(order, start=1)}
    unknown = sorted({v for v in text.dropna().unique() if v not in mapping})
    if unknown:
        raise ValueError(
            f"'{series.name}' has values missing from category_order: {unknown[:10]}"
            + (" …" if len(unknown) > 10 else "")
        )
    return pd.Series(
        [mapping[v] if isinstance(v, str) else np.nan for v in text],
        index=series.index,
        dtype=float,
    )


def _binary_encode(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Replace ``col`` with the bits of its ordinal code (``col_0`` is the most significant)."""
    codes = _ordinal_codes(df[col])
    n_bits = max(1, int(codes.max()).bit_length()) if codes.notna().any() else 1
    ints = codes.fillna(0).astype(int).to_numpy()
    bits = {f"{col}_{i}": (ints >> (n_bits - 1 - i)) & 1 for i in range(n_bits)}
    position = df.columns.get_loc(col)
    out = df.drop(columns=[col])
    for offset, (name, values) in enumerate(bits.items()):
        out.insert(position + offset, name, values)
    return out


def _target_encode(series: pd.Series, y: pd.Series, smoothing: float) -> pd.Series:
    """Smoothed mean of ``y`` per category; unseen or missing categories get the overall mean."""
    overall = float(y.mean())
    stats = y.groupby(series).agg(["mean", "count"])
    encoded = (stats["count"] * stats["mean"] + smoothing * overall) / (stats["count"] + smoothing)
    return series.map(encoded).astype(float).fillna(overall)


async def encode_categorical(
    dataset_version_id: str,
    columns: list[str],
    strategy: str,
    target_column: Optional[str] = None,
    drop_first: bool = True,
    smoothing: float = 1.0,
    category_order: Optional[list[str]] = None,
    *,
    session: Optional[AsyncSession] = None,
) -> tuple[str, str]:
    args = EncodeCategoricalArgs(
        dataset_version_id=dataset_version_id,
        columns=columns,
        strategy=strategy, # type: ignore[arg-type]
        target_column=target_column,
        drop_first=drop_first,
        smoothing=smoothing,
        category_order=category_order,
    )

    async def _run(db: AsyncSession) -> tuple[str, str]:
        parent = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(parent)
        _assert_columns_exist(df, args.columns)
        if args.strategy == EncodeStrategy.target and args.target_column:
            _assert_numeric_columns(df, [args.target_column])
        
        result = df.copy()

        if args.strategy == EncodeStrategy.one_hot:
            result = pd.get_dummies(result, columns=args.columns, drop_first=args.drop_first)
        elif args.strategy == EncodeStrategy.label:
            for col in args.columns:
                result[col] = _label_codes(result[col])
        elif args.strategy == EncodeStrategy.ordinal:
            for col in args.columns:
                result[col] = (
                    _ordered_codes(result[col], args.category_order)
                    if args.category_order else _ordinal_codes(result[col])
                )
        elif args.strategy == EncodeStrategy.frequency:
            for col in args.columns:
                freq = result[col].value_counts(normalize=True)
                result[f"{col}_freq"] = result[col].map(freq)
        elif args.strategy == EncodeStrategy.binary:
            for col in args.columns:
                result = _binary_encode(result, col)
        elif args.strategy == EncodeStrategy.target and args.target_column:
            y = result[args.target_column].astype(float)
            for col in args.columns:
                result[col] = _target_encode(result[col], y, args.smoothing)

        added = [c for c in result.columns if c not in df.columns]
        summary = (
            f"Encoded {len(args.columns)} column(s) using {args.strategy.value}"
            + (f"; added {len(added)} column(s)." if added else " in place.")
        )
        return await _persist_new_version(
            db,
            parent=parent,
            df=result,
            action_performed="encode_categorical",
            parameters_used={
                "strategy": args.strategy.value,
                "columns": args.columns,
                **({"drop_first": args.drop_first} if args.strategy == EncodeStrategy.one_hot else {}),
                **({"category_order": args.category_order}
                   if args.strategy == EncodeStrategy.ordinal and args.category_order else {}),
                **({"target_column": args.target_column, "smoothing": args.smoothing}
                   if args.strategy == EncodeStrategy.target else {}),
            },
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
    keep_original: bool = False,
    *,
    session: Optional[AsyncSession] = None,
) -> tuple[str, str]:
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import VarianceThreshold

    args = ReduceDimensionsArgs(
        dataset_version_id=dataset_version_id,
        columns=columns,
        strategy=strategy, # type: ignore[arg-type]
        # The model may send null for these; that means "use the default".
        **({"n_components": n_components} if n_components is not None else {}),
        **({"threshold": threshold} if threshold is not None else {}),
        keep_original=keep_original,
    )

    async def _run(db: AsyncSession) -> tuple[str, str]:
        parent = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(parent)
        _assert_numeric_columns(df, args.columns)
        result = df.copy()

        if args.strategy == ReduceDimensionsStrategy.pca:
            limit = min(len(args.columns), len(df))
            if args.n_components > limit:
                raise ValueError(f"n_components={args.n_components} is more than the {limit} the selected columns allow")
            pca = PCA(n_components=args.n_components)
            comps = pca.fit_transform(result[args.columns].fillna(0))
            for i in range(comps.shape[1]):
                result[f"pca_{i}"] = comps[:, i]
            if not args.keep_original:
                result = result.drop(columns=args.columns)
            explained = float(np.sum(pca.explained_variance_ratio_)) * 100
            summary = (
                f"PCA reduced {len(args.columns)} column(s) to {args.n_components} component(s) "
                f"explaining {explained:.1f}% of their variance."
            )
            parameters: dict[str, Any] = {
                "strategy": args.strategy.value,
                "columns": args.columns,
                "n_components": args.n_components,
                "keep_original": args.keep_original,
            }
        else:
            vt = VarianceThreshold(threshold=args.threshold)
            vt.fit(result[args.columns].fillna(0))
            to_keep = np.array(args.columns)[vt.get_support()]
            to_drop = [c for c in args.columns if c not in to_keep]
            result = result.drop(columns=to_drop)
            summary = (
                f"Variance threshold {args.threshold} dropped {len(to_drop)} of {len(args.columns)} column(s)"
                + (f": {', '.join(to_drop)}." if to_drop else ".")
            )
            parameters = {
                "strategy": args.strategy.value,
                "columns": args.columns,
                "threshold": args.threshold,
                "dropped_columns": to_drop,
            }

        return await _persist_new_version(
            db,
            parent=parent,
            df=result,
            action_performed="reduce_dimensions",
            parameters_used=parameters,
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
    interaction_only: bool = False,
    period: Optional[float] = None,
    *,
    session: Optional[AsyncSession] = None,
) -> tuple[str, str]:
    from sklearn.preprocessing import PolynomialFeatures
    args = FeatureEngineerArgs(
        dataset_version_id=dataset_version_id,
        columns=columns,
        strategy=strategy, # type: ignore[arg-type]
        degree=degree if degree is not None else 2,
        interaction_only=interaction_only,
        period=period,
    )

    async def _run(db: AsyncSession) -> tuple[str, str]:
        parent = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(parent)
        _assert_columns_exist(df, args.columns)
        result = df.copy()
        parameters: dict[str, Any] = {"strategy": args.strategy.value, "columns": args.columns}

        if args.strategy == FeatureEngineerStrategy.polynomial:
            _assert_numeric_columns(df, args.columns)
            parameters.update(degree=args.degree, interaction_only=args.interaction_only)
            poly = PolynomialFeatures(degree=args.degree, interaction_only=args.interaction_only, include_bias=False)
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
            _assert_numeric_columns(df, args.columns)
            if args.period is not None:
                parameters["period"] = args.period
            for col in args.columns:
                cycle = args.period if args.period is not None else result[col].max()
                if cycle > 0:
                    result[f"{col}_sin"] = np.sin(2 * np.pi * result[col] / cycle)
                    result[f"{col}_cos"] = np.cos(2 * np.pi * result[col] / cycle)
        elif args.strategy == FeatureEngineerStrategy.ratio:
            _assert_numeric_columns(df, args.columns)
            result[f"{args.columns[0]}_ratio_{args.columns[1]}"] = result[args.columns[0]] / result[args.columns[1]].replace(0, np.nan)

        added = [c for c in result.columns if c not in df.columns]
        summary = (
            f"Engineered {len(added)} feature(s) using {args.strategy.value} on {len(args.columns)} column(s)"
            + (f": {', '.join(added[:20])}{' …' if len(added) > 20 else ''}." if added else ".")
        )
        return await _persist_new_version(
            db,
            parent=parent,
            df=result,
            action_performed="feature_engineer",
            parameters_used=parameters,
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
    target_column: str = Field(..., min_length=1, description="Class label column to balance.")
    strategy: BalanceDataStrategy = Field(..., description="smote (oversample minority classes) or random_undersample.")
    random_state: int = Field(42, description="Random seed")
    k_neighbors: int = Field(5, description="Neighbours SMOTE uses to synthesise new samples.", ge=1)

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        return _validate_uuid_str(value)


_MAX_BALANCE_CLASSES = 50


def _format_counts(counts: pd.Series) -> str:
    return ", ".join(f"{label}: {int(n)}" for label, n in counts.sort_index().items())


def _smote(
    df: pd.DataFrame,
    target: str,
    counts: pd.Series,
    k_neighbors: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Oversample every class up to the largest one with SMOTE.

    Each synthetic row lies between a minority row and one of its k nearest
    same-class neighbours in the numeric columns; other columns are copied
    from the minority row.
    """
    from sklearn.neighbors import NearestNeighbors

    numeric = [c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric:
        raise ValueError("SMOTE needs at least one numeric feature column")
    largest = int(counts.max())
    synthetic: list[pd.DataFrame] = []
    for label, size in counts.items():
        need = largest - int(size)
        if need == 0:
            continue
        group = df[df[target] == label]
        if len(group) <= k_neighbors:
            raise ValueError(
                f"Class {label!r} has {len(group)} rows; SMOTE needs more rows than "
                f"k_neighbors ({k_neighbors}). Lower k_neighbors or use random_undersample."
            )
        X = group[numeric].astype(float)
        X = X.fillna(X.mean()).fillna(0).to_numpy()
        neighbours = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
        base = rng.integers(0, len(group), need)
        partner = neighbours[base, rng.integers(0, k_neighbors, need)]
        rows = group.iloc[base].copy()
        rows[numeric] = X[base] + rng.random((need, 1)) * (X[partner] - X[base])
        synthetic.append(rows)
    return pd.concat([df, *synthetic], ignore_index=True)


async def balance_data(
    dataset_version_id: str,
    target_column: str,
    strategy: str,
    random_state: int = 42,
    k_neighbors: int = 5,
    *,
    session: Optional[AsyncSession] = None,
) -> tuple[str, str]:
    args = BalanceDataArgs(
        dataset_version_id=dataset_version_id,
        target_column=target_column,
        strategy=strategy, # type: ignore[arg-type]
        random_state=random_state,
        k_neighbors=k_neighbors,
    )

    async def _run(db: AsyncSession) -> tuple[str, str]:
        parent = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(parent)
        _assert_columns_exist(df, [args.target_column])
        labelled = df[df[args.target_column].notna()]
        counts = labelled[args.target_column].value_counts()
        if len(counts) < 2:
            raise ValueError(f"'{args.target_column}' needs at least 2 classes to balance")
        if len(counts) > _MAX_BALANCE_CLASSES:
            raise ValueError(
                f"'{args.target_column}' has {len(counts)} distinct values; balancing needs a "
                "class label column, not a continuous target"
            )

        rng = np.random.default_rng(args.random_state)
        if args.strategy == BalanceDataStrategy.smote:
            result = _smote(labelled, args.target_column, counts, args.k_neighbors, rng)
        else:
            result = (
                labelled.groupby(args.target_column, group_keys=False)
                .sample(n=int(counts.min()), random_state=args.random_state)
                .sort_index()
                .reset_index(drop=True)
            )

        after = result[args.target_column].value_counts()
        summary = (
            f"Balanced '{args.target_column}' with {args.strategy.value}: rows {len(df)} → {len(result)}; "
            f"class counts {_format_counts(counts)} → {_format_counts(after)}."
        )
        return await _persist_new_version(
            db,
            parent=parent,
            df=result,
            action_performed="balance_data",
            parameters_used={
                "strategy": args.strategy.value,
                "target_column": args.target_column,
                "random_state": args.random_state,
                **({"k_neighbors": args.k_neighbors} if args.strategy == BalanceDataStrategy.smote else {}),
            },
            summary_message=summary,
        )

    if session is not None:
        return await _run(session)
    async with AsyncSessionLocal() as db:
        return await _run(db)


# ---------------------------------------------------------------------------
# Data inspection: head / tail / info / describe / shape / value counts /
# missing values / correlation / profiling report
# ---------------------------------------------------------------------------


class InspectView(str, Enum):
    head = "head"
    tail = "tail"
    info = "info"
    describe = "describe"
    shape = "shape"
    value_counts = "value_counts"
    missing = "missing"
    correlation = "correlation"
    profile = "profile"


_MAX_INSPECT_ROWS = 50


class InspectDataArgs(BaseModel):
    dataset_version_id: str = Field(..., min_length=36, max_length=36)
    view: InspectView = Field(
        ...,
        description=(
            "head / tail: the first or last n_rows rows; info: dtype, non-null, missing and memory "
            "per column; describe: count, mean, std, min, quartiles, max (numeric) and unique, top, "
            "freq (text); shape: row and column counts; value_counts: the top_n most frequent values "
            "per column; missing: missing values per column (isnull, including text such as 'NA' or "
            "'?'); correlation: correlation matrix of the numeric columns (method); profile: a "
            "profiling report per column (type, missing, unique, zeros, mean, std, min, median, max, "
            "skew, kurtosis, most frequent value)."
        ),
    )
    columns: Optional[list[str]] = Field(
        None,
        description="Columns to include. Omit to use every column (every numeric one for correlation).",
    )
    n_rows: int = Field(10, description="Rows shown by head and tail.", ge=1, le=_MAX_INSPECT_ROWS)
    top_n: int = Field(10, description="Most frequent values listed per column by value_counts.", ge=1, le=50)
    method: CorrelationMethod = Field(
        CorrelationMethod.pearson,
        description="Correlation coefficient for the correlation view: pearson, spearman or kendall.",
    )

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        return _validate_uuid_str(value)

    @field_validator("columns")
    @classmethod
    def _optional_names(cls, value: Optional[list[str]]) -> Optional[list[str]]:
        return _clean_optional_names(value)


def _num(value: Any, digits: int = 6) -> Optional[float]:
    """A rounded float, or None for missing / non-finite values."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return round(f, digits) if np.isfinite(f) else None


def _json_cell(value: Any) -> Any:
    """A strict-JSON scalar: NaN / NA become None, timestamps and other objects text."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _num(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _uniform_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every row gets every key (in first-seen order), so tables keep all their columns."""
    keys: list[str] = []
    for row in rows:
        keys.extend(k for k in row if k not in keys)
    return [{k: _json_cell(row.get(k)) for k in keys} for row in rows]


def _is_plain_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _positioned(df: pd.DataFrame, start: int) -> list[dict[str, Any]]:
    label = "row" if "row" not in df.columns else "_row"
    out = df.reset_index(drop=True)
    out.insert(0, label, range(start, start + len(out)))
    return out.to_dict(orient="records")


def _inspect(df: pd.DataFrame, args: InspectDataArgs) -> tuple[str, list[dict[str, Any]]]:
    n_rows, n_cols = df.shape
    view = args.view
    if view == InspectView.head:
        return f"head — first {min(args.n_rows, n_rows)} of {n_rows} rows", _positioned(df.head(args.n_rows), 0)
    if view == InspectView.tail:
        part = df.tail(args.n_rows)
        return f"tail — last {len(part)} of {n_rows} rows", _positioned(part, n_rows - len(part))
    if view == InspectView.shape:
        return "shape", [{"rows": n_rows, "columns": n_cols}]

    if view == InspectView.correlation:
        numeric = [c for c in df.columns if _is_plain_numeric(df[c])]
        if len(numeric) < 2:
            raise ValueError("The correlation view needs at least 2 numeric columns")
        corr = df[numeric].corr(method=args.method.value)
        rows = [{"column": c, **{o: _num(corr.loc[c, o], 4) for o in numeric}} for c in numeric]
        return f"correlation ({args.method.value})", rows

    if view == InspectView.describe:
        desc = df.describe(include="all").T
        rows = [{"column": str(col), **{str(k): v for k, v in stats.items()}} for col, stats in desc.iterrows()]
        return f"describe — {n_rows} rows", rows

    if view == InspectView.value_counts:
        rows = []
        for col in df.columns:
            counts = df[col].value_counts(dropna=False)
            for value, count in counts.head(args.top_n).items():
                rows.append({
                    "column": col,
                    "value": "(missing)" if _json_cell(value) is None else _json_cell(value),
                    "count": int(count),
                    "percent": round(100 * int(count) / n_rows, 2) if n_rows else 0.0,
                    "distinct_values": int(len(counts)),
                })
        return f"value_counts — top {args.top_n} per column", rows

    # info, missing and profile count text placeholders ("NA", "?", blanks) as missing,
    # the way remove_missing_data treats them.
    clean = _coerce_fake_nulls(df)
    missing = clean.isna().sum()
    if view == InspectView.info:
        memory = df.memory_usage(deep=True)
        rows = [
            {
                "column": c,
                "dtype": str(df[c].dtype),
                "non_null": int(n_rows - missing[c]),
                "missing": int(missing[c]),
                "memory_kb": round(float(memory[c]) / 1024, 2),
            }
            for c in df.columns
        ]
        return f"info — {n_rows} rows × {n_cols} columns, {float(memory.sum()) / 1024:.1f} KB", rows
    if view == InspectView.missing:
        rows = [
            {
                "column": c,
                "missing": int(missing[c]),
                "missing_pct": round(100 * float(missing[c]) / n_rows, 2) if n_rows else 0.0,
                "non_null": int(n_rows - missing[c]),
            }
            for c in df.columns
        ]
        rows.sort(key=lambda r: r["missing"], reverse=True)
        return f"missing values — {int(missing.sum())} missing cell(s) in {n_rows} rows", rows

    rows = []
    for c in df.columns:
        s = clean[c]
        row: dict[str, Any] = {
            "column": c,
            "dtype": str(df[c].dtype),
            "non_null": int(n_rows - missing[c]),
            "missing": int(missing[c]),
            "missing_pct": round(100 * float(missing[c]) / n_rows, 2) if n_rows else 0.0,
            "unique": int(s.nunique(dropna=True)),
        }
        if _is_plain_numeric(s):
            x = s.astype(float)
            row.update(
                zeros=int((x == 0).sum()),
                mean=_num(x.mean()),
                std=_num(x.std()),
                min=_num(x.min()),
                median=_num(x.median()),
                max=_num(x.max()),
                skew=_num(x.skew(), 4),
                kurtosis=_num(x.kurt(), 4),
            )
        counts = s.value_counts()
        if len(counts):
            row.update(top=counts.index[0], top_freq=int(counts.iloc[0]))
        rows.append(row)
    return f"profile — {n_rows} rows × {n_cols} columns", rows


async def inspect_data(
    dataset_version_id: str,
    view: str,
    columns: Optional[list[str]] = None,
    n_rows: int = 10,
    top_n: int = 10,
    method: str = "pearson",
    *,
    session: Optional[AsyncSession] = None,
) -> dict[str, Any]:
    """Look at the data the way pandas does (head, info, describe, …); never modifies it."""
    args = InspectDataArgs(
        dataset_version_id=dataset_version_id,
        view=view,  # type: ignore[arg-type]
        columns=columns,
        n_rows=n_rows,
        top_n=top_n,
        method=method,  # type: ignore[arg-type]
    )

    async def _run(db: AsyncSession) -> dict[str, Any]:
        version = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(version)
        if args.columns:
            _assert_columns_exist(df, args.columns)
            df = df[args.columns]
        title, rows = _inspect(df, args)
        return {
            "dataset_version_id": args.dataset_version_id,
            "view": args.view.value,
            "title": title,
            "rows": _uniform_rows(rows),
        }

    if session is not None:
        return await _run(session)
    async with AsyncSessionLocal() as db:
        return await _run(db)


# ---------------------------------------------------------------------------
# Row filtering, duplicates, type conversion and text cleaning
# ---------------------------------------------------------------------------


class DropRowsMode(str, Enum):
    condition = "condition"
    row_range = "row_range"


class RowOperator(str, Enum):
    equals = "equals"
    not_equals = "not_equals"
    greater_than = "greater_than"
    greater_or_equal = "greater_or_equal"
    less_than = "less_than"
    less_or_equal = "less_or_equal"
    contains = "contains"
    is_missing = "is_missing"
    not_missing = "not_missing"


_VALUELESS_OPERATORS = frozenset({RowOperator.is_missing, RowOperator.not_missing})
_OPERATOR_SYMBOLS = {
    RowOperator.equals: "==",
    RowOperator.not_equals: "!=",
    RowOperator.greater_than: ">",
    RowOperator.greater_or_equal: ">=",
    RowOperator.less_than: "<",
    RowOperator.less_or_equal: "<=",
    RowOperator.contains: "contains",
    RowOperator.is_missing: "is missing",
    RowOperator.not_missing: "is not missing",
}


class DropRowsArgs(BaseModel):
    dataset_version_id: str = Field(..., min_length=36, max_length=36)
    mode: DropRowsMode = Field(
        ...,
        description=(
            "condition: the rows where `column` `operator` `value` holds; row_range: the rows at "
            "positions start_row (inclusive) to end_row (exclusive), counted from 0 like pandas iloc."
        ),
    )
    column: Optional[str] = Field(None, description="Column the condition tests (condition mode).")
    operator: RowOperator = Field(
        RowOperator.equals,
        description=(
            "Comparison for condition mode: equals, not_equals, greater_than, greater_or_equal, "
            "less_than, less_or_equal, contains (text, case-insensitive), is_missing, not_missing."
        ),
    )
    value: Optional[str] = Field(
        None,
        description="Value the column is compared with; a number for numeric columns. Not used by is_missing / not_missing.",
    )
    start_row: int = Field(0, description="First row position of the range (row_range, 0-based, inclusive).", ge=0)
    end_row: Optional[int] = Field(
        None,
        description="Row position where the range stops (row_range, exclusive). Omit to go to the last row.",
        ge=1,
    )
    keep_matching: bool = Field(
        False,
        description="Keep only the matching rows and drop all others (a filter), instead of dropping the matching rows.",
    )

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        return _validate_uuid_str(value)

    @field_validator("column", mode="before")
    @classmethod
    def _blank_column(cls, value: Any) -> Any:
        return value.strip() or None if isinstance(value, str) else value

    @field_validator("value", mode="before")
    @classmethod
    def _value_as_text(cls, value: Any) -> Any:
        return None if value is None else str(value)

    @model_validator(mode="after")
    def _mode_needs(self) -> "DropRowsArgs":
        if self.mode == DropRowsMode.condition:
            if not self.column:
                raise ValueError("condition mode needs column")
            if self.operator not in _VALUELESS_OPERATORS and (self.value is None or self.value == ""):
                raise ValueError(f"The {self.operator.value} operator needs a value")
        elif self.end_row is not None and self.end_row <= self.start_row:
            raise ValueError("end_row must be larger than start_row")
        return self


def _condition_mask(df: pd.DataFrame, args: DropRowsArgs) -> pd.Series:
    series = df[args.column]
    op = args.operator
    if op == RowOperator.is_missing:
        return _coerce_fake_nulls(df[[args.column]])[args.column].isna()
    if op == RowOperator.not_missing:
        return _coerce_fake_nulls(df[[args.column]])[args.column].notna()
    if op == RowOperator.contains:
        return series.astype("string").str.contains(args.value, case=False, regex=False, na=False).astype(bool)

    text = args.value.strip()
    if pd.api.types.is_bool_dtype(series):
        target: Any = text.lower() in {"true", "1", "yes", "t", "y"}
    elif pd.api.types.is_numeric_dtype(series):
        try:
            target = float(text)
        except ValueError as exc:
            raise ValueError(f"'{args.column}' is numeric, so value must be a number (got {args.value!r})") from exc
    elif pd.api.types.is_datetime64_any_dtype(series):
        try:
            target = pd.Timestamp(text)
        except ValueError as exc:
            raise ValueError(f"'{args.column}' holds dates, so value must be a date (got {args.value!r})") from exc
    else:
        series = series.astype("string")
        target = args.value
    compare = {
        RowOperator.equals: series == target,
        RowOperator.not_equals: series != target,
        RowOperator.greater_than: series > target,
        RowOperator.greater_or_equal: series >= target,
        RowOperator.less_than: series < target,
        RowOperator.less_or_equal: series <= target,
    }[op]
    return compare.fillna(False).astype(bool)


async def drop_rows(
    dataset_version_id: str,
    mode: str,
    column: Optional[str] = None,
    operator: str = "equals",
    value: Optional[str] = None,
    start_row: int = 0,
    end_row: Optional[int] = None,
    keep_matching: bool = False,
    *,
    session: Optional[AsyncSession] = None,
) -> tuple[str, str]:
    """Drop (or keep only) the rows matching a condition or a position range."""
    args = DropRowsArgs(
        dataset_version_id=dataset_version_id,
        mode=mode,  # type: ignore[arg-type]
        column=column,
        operator=operator,  # type: ignore[arg-type]
        value=value,
        start_row=start_row,
        end_row=end_row,
        keep_matching=keep_matching,
    )

    async def _run(db: AsyncSession) -> tuple[str, str]:
        parent = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(parent)
        n = len(df)
        if args.mode == DropRowsMode.condition:
            _assert_columns_exist(df, [args.column])
            matching = _condition_mask(df, args)
            rule = f"{args.column} {_OPERATOR_SYMBOLS[args.operator]}"
            if args.operator not in _VALUELESS_OPERATORS:
                rule += f" {args.value}"
            parameters: dict[str, Any] = {
                "mode": args.mode.value, "column": args.column, "operator": args.operator.value,
                **({"value": args.value} if args.operator not in _VALUELESS_OPERATORS else {}),
            }
        else:
            end = min(args.end_row if args.end_row is not None else n, n)
            positions = np.zeros(n, dtype=bool)
            positions[args.start_row:end] = True
            matching = pd.Series(positions, index=df.index)
            rule = (
                f"rows {args.start_row}–{end - 1}" if end > args.start_row
                else f"rows from position {args.start_row} (the dataset has {n} rows)"
            )
            parameters = {"mode": args.mode.value, "start_row": args.start_row, "end_row": args.end_row}
        parameters["keep_matching"] = args.keep_matching

        drop = ~matching if args.keep_matching else matching
        n_drop = int(drop.sum())
        if n_drop == 0:
            return str(parent.id), f"No rows to drop ({'keeping' if args.keep_matching else 'matching'} {rule}). No changes made."
        if n_drop == n:
            raise ValueError(f"This would drop all {n} rows ({rule}); nothing would be left.")
        result = df.loc[~drop].reset_index(drop=True)
        verb = "Kept only rows where" if args.keep_matching else "Dropped rows where"
        if args.mode == DropRowsMode.row_range:
            verb = "Kept only" if args.keep_matching else "Dropped"
        return await _persist_new_version(
            db,
            parent=parent,
            df=result,
            action_performed="drop_rows",
            parameters_used={**parameters, "rows_dropped": n_drop},
            summary_message=f"{verb} {rule}: {n_drop} row(s) dropped, rows {n} → {len(result)}.",
        )

    if session is not None:
        return await _run(session)
    async with AsyncSessionLocal() as db:
        return await _run(db)


class DuplicateKeep(str, Enum):
    first = "first"
    last = "last"
    none = "none"


class DropDuplicatesArgs(BaseModel):
    dataset_version_id: str = Field(..., min_length=36, max_length=36)
    columns: Optional[list[str]] = Field(
        None,
        description="Rows count as duplicates when these columns match. Omit to compare whole rows.",
    )
    keep: DuplicateKeep = Field(
        DuplicateKeep.first,
        description="Which copy of a duplicated row to keep: first, last, or none (drop every copy).",
    )

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        return _validate_uuid_str(value)

    @field_validator("columns")
    @classmethod
    def _optional_names(cls, value: Optional[list[str]]) -> Optional[list[str]]:
        return _clean_optional_names(value)


async def drop_duplicates(
    dataset_version_id: str,
    columns: Optional[list[str]] = None,
    keep: str = "first",
    *,
    session: Optional[AsyncSession] = None,
) -> tuple[str, str]:
    """Drop duplicated rows (pandas drop_duplicates)."""
    args = DropDuplicatesArgs(dataset_version_id=dataset_version_id, columns=columns, keep=keep)  # type: ignore[arg-type]

    async def _run(db: AsyncSession) -> tuple[str, str]:
        parent = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(parent)
        if args.columns:
            _assert_columns_exist(df, args.columns)
        keep_arg: Any = False if args.keep == DuplicateKeep.none else args.keep.value
        result = df.drop_duplicates(subset=args.columns, keep=keep_arg).reset_index(drop=True)
        removed = len(df) - len(result)
        scope = f" on columns {args.columns}" if args.columns else ""
        if removed == 0:
            return str(parent.id), f"No duplicate rows found{scope}. No changes made."
        return await _persist_new_version(
            db,
            parent=parent,
            df=result,
            action_performed="drop_duplicates",
            parameters_used={"columns": args.columns, "keep": args.keep.value, "rows_dropped": removed},
            summary_message=(
                f"Dropped {removed} duplicate row(s){scope} (keep={args.keep.value}); "
                f"rows {len(df)} → {len(result)}."
            ),
        )

    if session is not None:
        return await _run(session)
    async with AsyncSessionLocal() as db:
        return await _run(db)


class TargetType(str, Enum):
    FLOAT = "float"
    INTEGER = "integer"
    STRING = "string"
    CATEGORY = "category"
    BOOLEAN = "boolean"
    DATETIME = "datetime"


class ConvertTypesArgs(BaseModel):
    dataset_version_id: str = Field(..., min_length=36, max_length=36)
    columns: list[str] = Field(..., min_length=1, description="Columns to convert.")
    target_type: TargetType = Field(
        ...,
        description=(
            "New type: float, integer (whole numbers), string (text), category, boolean "
            "(true/false, yes/no, 1/0) or datetime (parses dates and times)."
        ),
    )
    errors: str = Field(
        "coerce",
        description="coerce: values that cannot be converted become missing; raise: stop with an error instead.",
        json_schema_extra={"enum": ["coerce", "raise"]},
    )
    datetime_format: Optional[str] = Field(
        None,
        description="For datetime: strftime format such as '%d/%m/%Y %H:%M'. Omit to detect it per value.",
    )

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        return _validate_uuid_str(value)

    @field_validator("columns")
    @classmethod
    def _non_empty_names(cls, value: list[str]) -> list[str]:
        return _clean_names(value)

    @field_validator("errors")
    @classmethod
    def _valid_errors(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in {"coerce", "raise"}:
            raise ValueError("errors must be 'coerce' or 'raise'")
        return value

    @field_validator("datetime_format", mode="before")
    @classmethod
    def _blank_format(cls, value: Any) -> Any:
        return value.strip() or None if isinstance(value, str) else value


_TRUE_TEXT = frozenset({"true", "t", "yes", "y", "1", "on"})
_FALSE_TEXT = frozenset({"false", "f", "no", "n", "0", "off"})


def _to_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")
    if pd.api.types.is_numeric_dtype(series):
        return series.map(lambda v: pd.NA if pd.isna(v) else bool(v != 0)).astype("boolean")

    def read(v: Any) -> Any:
        if not isinstance(v, str):
            return pd.NA
        text = v.strip().lower()
        return True if text in _TRUE_TEXT else False if text in _FALSE_TEXT else pd.NA

    return series.map(read).astype("boolean")


def _convert_column(series: pd.Series, args: ConvertTypesArgs) -> pd.Series:
    kind = args.target_type
    if kind in (TargetType.FLOAT, TargetType.INTEGER):
        text = series.map(lambda v: v.strip() if isinstance(v, str) else v)
        numbers = pd.to_numeric(text, errors=args.errors)
        if kind == TargetType.FLOAT:
            return numbers.astype("float64")
        whole = numbers.dropna()
        if len(whole) and not (whole % 1 == 0).all():
            example = whole[whole % 1 != 0].iloc[0]
            raise ValueError(f"'{series.name}' has fractional values (e.g. {example}); convert it to float instead")
        return numbers.astype("int64" if numbers.notna().all() else "Int64")
    if kind == TargetType.STRING:
        return series.astype("string")
    if kind == TargetType.CATEGORY:
        return series.astype("category")
    if kind == TargetType.BOOLEAN:
        flags = _to_boolean(series)
        bad = series.notna() & flags.isna()
        if bad.any() and args.errors == "raise":
            raise ValueError(f"'{series.name}' has values that are not true/false, e.g. {series[bad].iloc[0]!r}")
        return flags.astype(bool) if flags.notna().all() else flags
    if _is_plain_numeric(series) and not args.datetime_format:
        return pd.to_datetime(series, errors=args.errors)
    return pd.to_datetime(series, errors=args.errors, format=args.datetime_format or "mixed")


async def convert_types(
    dataset_version_id: str,
    columns: list[str],
    target_type: str,
    errors: str = "coerce",
    datetime_format: Optional[str] = None,
    *,
    session: Optional[AsyncSession] = None,
) -> tuple[str, str]:
    """Cast columns to another type (astype / to_numeric / to_datetime)."""
    args = ConvertTypesArgs(
        dataset_version_id=dataset_version_id,
        columns=columns,
        target_type=target_type,  # type: ignore[arg-type]
        errors=errors,
        datetime_format=datetime_format,
    )

    async def _run(db: AsyncSession) -> tuple[str, str]:
        parent = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(parent)
        _assert_columns_exist(df, args.columns)
        # Text placeholders ("NA", "?", blanks) are missing values, not conversion failures.
        clean = _coerce_fake_nulls(df[args.columns])
        result = df.copy()
        lost: dict[str, int] = {}
        for col in args.columns:
            try:
                converted = _convert_column(clean[col], args)
            except (ValueError, TypeError) as exc:
                message = str(exc)
                raise ValueError(message if f"'{col}'" in message else f"'{col}': {message}") from exc
            failed = int((clean[col].notna() & converted.isna()).sum())
            if failed:
                lost[col] = failed
            result[col] = converted
        summary = f"Converted {len(args.columns)} column(s) to {args.target_type.value}: {', '.join(args.columns)}."
        if lost:
            summary += " Values that could not be converted are now missing: " + ", ".join(
                f"{c} ({n})" for c, n in lost.items()
            ) + "."
        return await _persist_new_version(
            db,
            parent=parent,
            df=result,
            action_performed="convert_types",
            parameters_used={
                "columns": args.columns,
                "target_type": args.target_type.value,
                "errors": args.errors,
                **({"datetime_format": args.datetime_format} if args.datetime_format else {}),
                "values_made_missing": lost,
            },
            summary_message=summary,
        )

    if session is not None:
        return await _run(session)
    async with AsyncSessionLocal() as db:
        return await _run(db)


class TextOperation(str, Enum):
    strip = "strip"
    lower = "lower"
    upper = "upper"
    title = "title"
    collapse_whitespace = "collapse_whitespace"
    replace = "replace"


class CleanTextArgs(BaseModel):
    dataset_version_id: str = Field(..., min_length=36, max_length=36)
    columns: list[str] = Field(..., min_length=1, description="Text columns to clean.")
    operation: TextOperation = Field(
        ...,
        description=(
            "strip (trim leading/trailing spaces), lower, upper, title (Title Case), "
            "collapse_whitespace (runs of spaces become one, then trim), or replace (find → replace_with)."
        ),
    )
    find: Optional[str] = Field(None, description="For replace: the text (or regular expression) to find.", max_length=200)
    replace_with: str = Field("", description="For replace: what each match becomes (empty to delete it).", max_length=200)
    use_regex: bool = Field(False, description="For replace: treat find as a regular expression.")

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        return _validate_uuid_str(value)

    @field_validator("columns")
    @classmethod
    def _non_empty_names(cls, value: list[str]) -> list[str]:
        return _clean_names(value)

    @model_validator(mode="after")
    def _replace_needs_find(self) -> "CleanTextArgs":
        if self.operation == TextOperation.replace:
            if not self.find:
                raise ValueError("replace needs find")
            if self.use_regex:
                try:
                    re.compile(self.find)
                except re.error as exc:
                    raise ValueError(f"find is not a valid regular expression: {exc}") from exc
        return self


def _clean_strings(values: pd.Series, args: CleanTextArgs) -> pd.Series:
    text = values.str
    op = args.operation
    if op == TextOperation.strip:
        return text.strip()
    if op == TextOperation.lower:
        return text.lower()
    if op == TextOperation.upper:
        return text.upper()
    if op == TextOperation.title:
        return text.title()
    if op == TextOperation.collapse_whitespace:
        return text.replace(r"\s+", " ", regex=True).str.strip()
    return text.replace(args.find, args.replace_with, regex=args.use_regex)


async def clean_text(
    dataset_version_id: str,
    columns: list[str],
    operation: str,
    find: Optional[str] = None,
    replace_with: str = "",
    use_regex: bool = False,
    *,
    session: Optional[AsyncSession] = None,
) -> tuple[str, str]:
    """Strip, re-case, collapse spaces in, or find-and-replace text columns."""
    args = CleanTextArgs(
        dataset_version_id=dataset_version_id,
        columns=columns,
        operation=operation,  # type: ignore[arg-type]
        find=find,
        replace_with=replace_with,
        use_regex=use_regex,
    )

    async def _run(db: AsyncSession) -> tuple[str, str]:
        parent = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(parent)
        _assert_columns_exist(df, args.columns)
        not_text = [
            c for c in args.columns
            if not (pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_object_dtype(df[c])
                    or isinstance(df[c].dtype, pd.CategoricalDtype))
        ]
        if not_text:
            raise ValueError(f"Not text columns: {not_text}. Convert them with convert_types (string) first.")

        result = df.copy()
        changed = 0
        for col in args.columns:
            original = df[col]
            was_category = isinstance(original.dtype, pd.CategoricalDtype)
            values = original.astype(object)
            is_text = values.map(lambda v: isinstance(v, str))
            before = values[is_text].astype(str)
            after = _clean_strings(before, args)
            changed += int((before != after).sum())
            cleaned = values.copy()
            cleaned[is_text] = after
            result[col] = cleaned.astype("category") if was_category else cleaned.astype(original.dtype)
        detail = (
            f" '{args.find}' → '{args.replace_with}'" if args.operation == TextOperation.replace else ""
        )
        return await _persist_new_version(
            db,
            parent=parent,
            df=result,
            action_performed="clean_text",
            parameters_used={
                "columns": args.columns,
                "operation": args.operation.value,
                **({"find": args.find, "replace_with": args.replace_with, "use_regex": args.use_regex}
                   if args.operation == TextOperation.replace else {}),
                "values_changed": changed,
            },
            summary_message=(
                f"Applied {args.operation.value}{detail} to {len(args.columns)} column(s); "
                f"{changed} value(s) changed."
            ),
        )

    if session is not None:
        return await _run(session)
    async with AsyncSessionLocal() as db:
        return await _run(db)


# ---------------------------------------------------------------------------
# Univariate feature selection (SelectKBest)
# ---------------------------------------------------------------------------


class KBestScore(str, Enum):
    f_regression = "f_regression"
    mutual_info_regression = "mutual_info_regression"


class SelectKBestArgs(BaseModel):
    dataset_version_id: str = Field(..., min_length=36, max_length=36)
    target_column: str = Field(..., description="Name of the target/response column.")
    feature_columns: Optional[list[str]] = Field(
        None,
        description="Candidate feature columns. Omit to consider every numeric column except the target.",
    )
    k: int = Field(5, description="Number of best-scoring features to keep.", ge=1)
    score_func: KBestScore = Field(
        KBestScore.f_regression,
        description=(
            "Score of each feature against the target: f_regression (linear F-test) or "
            "mutual_info_regression (also catches non-linear dependence)."
        ),
    )
    random_state: int = Field(42, description="Random seed of mutual_info_regression.")

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        return _validate_uuid_str(value)

    @field_validator("feature_columns")
    @classmethod
    def _optional_names(cls, value: Optional[list[str]]) -> Optional[list[str]]:
        return _clean_optional_names(value)


async def select_k_best(
    dataset_version_id: str,
    target_column: str,
    feature_columns: Optional[list[str]] = None,
    k: int = 5,
    score_func: str = "f_regression",
    random_state: int = 42,
    *,
    session: Optional[AsyncSession] = None,
) -> tuple[str, str]:
    """Keep the k features that score best against the target (SelectKBest); drop the rest."""
    from sklearn.feature_selection import f_regression, mutual_info_regression

    args = SelectKBestArgs(
        dataset_version_id=dataset_version_id,
        target_column=target_column,
        feature_columns=feature_columns,
        k=k,
        score_func=score_func,  # type: ignore[arg-type]
        random_state=random_state,
    )

    async def _run(db: AsyncSession) -> tuple[str, str]:
        parent = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(parent)
        _assert_numeric_columns(df, [args.target_column])
        if args.feature_columns:
            feature_cols = [c for c in args.feature_columns if c != args.target_column]
            _assert_numeric_columns(df, feature_cols)
        else:
            feature_cols = [
                c for c in df.columns
                if c != args.target_column and _is_plain_numeric(df[c]) and not c.startswith("Predicted_")
            ]
        if len(feature_cols) <= args.k:
            return str(parent.id), (
                f"Only {len(feature_cols)} numeric feature column(s), not more than k={args.k}; nothing to drop. "
                "No changes made."
            )
        complete = df[[*feature_cols, args.target_column]].dropna()
        if len(complete) < 5:
            raise ValueError(f"Need at least 5 rows without missing values; found {len(complete)}")
        X = complete[feature_cols].astype(float).to_numpy()
        y = complete[args.target_column].astype(float).to_numpy()
        if args.score_func == KBestScore.f_regression:
            scores, _ = f_regression(X, y)
        else:
            scores = mutual_info_regression(X, y, random_state=args.random_state)
        scores = np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0, posinf=np.finfo(float).max)
        order = np.argsort(-scores, kind="stable")
        selected = [feature_cols[i] for i in order[: args.k]]
        dropped = [c for c in feature_cols if c not in selected]
        ranked = {feature_cols[i]: round(float(scores[i]), 4) for i in order}

        result = df.drop(columns=dropped)
        return await _persist_new_version(
            db,
            parent=parent,
            df=result,
            action_performed="select_k_best",
            parameters_used={
                "target_column": args.target_column,
                "k": args.k,
                "score_func": args.score_func.value,
                **({"random_state": args.random_state} if args.score_func == KBestScore.mutual_info_regression else {}),
                "scores": ranked,
                "selected_features": selected,
                "dropped_features": dropped,
            },
            summary_message=(
                f"SelectKBest ({args.score_func.value}) kept {len(selected)} features: "
                + ", ".join(f"{c} ({ranked[c]})" for c in selected)
                + f". Dropped {len(dropped)}: {', '.join(dropped)}."
            ),
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
            "Handle missing values using drop_rows, fill_mean, fill_median, fill_mode, "
            "forward_fill, backward_fill, knn (n_neighbors) or iterative imputation "
            "(max_iter, random_state), on all columns or only `columns`. "
            "Saves a new parquet dataset version and logs lineage. "
            "Returns (new_version_id, summary_message)."
        ),
        args_model=RemoveMissingDataArgs,
    ),
    _openai_tool_schema(
        name="remove_outliers",
        description=(
            "Handle outliers in selected numeric columns: z_score (absolute standard deviations, "
            "default 3) or iqr (IQR fence multiplier, default 1.5) either drop the outlier rows "
            "(action=drop) or cap the values at the fence (action=cap); winsorize caps each tail "
            "at a percentile (tail fraction, default 0.05); clip caps values to fixed "
            "lower_bound / upper_bound. Saves a new dataset version and logs lineage. "
            "Returns (new_version_id, summary_message)."
        ),
        args_model=RemoveOutliersArgs,
    ),
    _openai_tool_schema(
        name="normalize_data",
        description=(
            "Normalize selected numeric columns using scikit-learn strategies "
            "standard, min_max (feature_min..feature_max), robust "
            "(quantile_low..quantile_high percentiles), or maxabs. Saves a new dataset version and logs "
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
        description=(
            "Apply non-linear transformations (log, box_cox, yeo_johnson, sqrt), binning "
            "(bin_equal_width / bin_equal_freq with n_bins) or binarize (threshold). "
            "Adds the results as new columns."
        ),
        args_model=TransformFeaturesArgs,
    ),
    _openai_tool_schema(
        name="encode_categorical",
        description=(
            "Encode categorical features using one_hot, label (0..n-1 codes), ordinal (1..n codes, "
            "in sorted order or a given category_order), frequency, target, or binary."
        ),
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
            "Use Recursive Feature Elimination (RFE) with linear regression, Ridge or Lasso to automatically "
            "select the most important features (from feature_columns, or all numeric columns) and drop the rest. "
            "Saves a new dataset version and logs lineage."
        ),
        args_model=SelectFeaturesRFEArgs,
    ),
    _openai_tool_schema(
        name="detect_anomalies",
        description=(
            "Detect multivariate anomalies using Isolation Forest on the numeric `columns` (default all). "
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
    _openai_tool_schema(
        name="inspect_data",
        description=(
            "Look at the data like pandas does, without changing it: head / tail (first or last "
            "n_rows rows), info (dtypes, non-null counts, memory), describe, shape, value_counts, "
            "missing (isnull counts per column), correlation (corr matrix) or profile (a full "
            "per-column profiling report). Returns a table."
        ),
        args_model=InspectDataArgs,
    ),
    _openai_tool_schema(
        name="drop_rows",
        description=(
            "Drop rows that match a condition on one column (equals, greater_than, contains, "
            "is_missing, …) or that sit in a row-position range; keep_matching=true keeps only "
            "those rows instead (a filter). Saves a new dataset version and logs lineage."
        ),
        args_model=DropRowsArgs,
    ),
    _openai_tool_schema(
        name="drop_duplicates",
        description=(
            "Drop duplicated rows, comparing whole rows or only `columns`, keeping the first, "
            "the last or no copy. Saves a new dataset version and logs lineage."
        ),
        args_model=DropDuplicatesArgs,
    ),
    _openai_tool_schema(
        name="convert_types",
        description=(
            "Cast columns to float, integer, string, category, boolean or datetime (type "
            "conversion and datetime parsing, with an optional datetime_format). Unconvertible "
            "values become missing (errors=coerce) or stop the tool (errors=raise). Saves a new "
            "dataset version and logs lineage."
        ),
        args_model=ConvertTypesArgs,
    ),
    _openai_tool_schema(
        name="clean_text",
        description=(
            "Clean text columns: strip spaces, lower / upper / title case, collapse repeated "
            "whitespace, or find-and-replace (plain text or regex). Saves a new dataset version "
            "and logs lineage."
        ),
        args_model=CleanTextArgs,
    ),
    _openai_tool_schema(
        name="select_k_best",
        description=(
            "SelectKBest: score every candidate feature against the target with f_regression or "
            "mutual_info_regression, keep the k best and drop the rest. Saves a new dataset "
            "version and logs lineage."
        ),
        args_model=SelectKBestArgs,
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
    "inspect_data": inspect_data,
    # Rows, types and text
    "drop_rows": drop_rows,
    "drop_duplicates": drop_duplicates,
    "convert_types": convert_types,
    "clean_text": clean_text,
    "select_k_best": select_k_best,
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
    "inspect_data",
    "drop_rows",
    "drop_duplicates",
    "convert_types",
    "clean_text",
    "select_k_best",
]
