"""Soft-sensor training and validation tools for Groq OpenAI-compatible calling.

OLS / PLS regression models are fit on dataset versions, persisted under the
storage key ``{owner}/{dataset}/models/{model_id}.pkl``, and validated with parity plots for
process soft-sensor performance assessment.
"""

from __future__ import annotations

import json
import math
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pydantic import BaseModel, Field, field_validator, model_validator
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    LeaveOneGroupOut,
    LeaveOneOut,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
    TimeSeriesSplit,
    train_test_split,
)
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import AsyncSessionLocal
from backend.mcp_server.eda_tools import (
    _assert_numeric_columns,
    _get_version,
    _load_dataframe,
    _openai_tool_schema,
    _persist_new_version,
    _validate_uuid_str,
)
from backend.mcp_server.model_registry import (
    lineage_models,
    register_model,
    resolve_model,
)
from backend.models import Dataset
from backend.storage import get_storage, model_key

_RANDOM_STATE = 42
_TEST_SIZE = 0.2
_MAX_LOO_ROWS = 1000

_MAX_CV_FITS = 1000

_SPLIT_STRATEGIES = ("random", "stratified", "sequential", "group")
_SPLIT_ALIASES = {
    "shuffle": "random",
    "shuffled": "random",
    "stratify": "stratified",
    "chronological": "sequential",
    "time_series": "sequential",
    "timeseries": "sequential",
    "temporal": "sequential",
    "ordered": "sequential",
    "grouped": "group",
    "group_shuffle": "group",
}
_CV_STRATEGIES = (
    "none", "kfold", "stratified_kfold", "leave_one_out", "time_series",
    "group_kfold", "leave_one_group_out", "repeated_kfold", "repeated_stratified_kfold",
    "shuffle_split", "stratified_shuffle_split", "walk_forward", "bootstrap",
)
_CV_ALIASES = {
    "": "none",
    "no": "none",
    "off": "none",
    "k_fold": "kfold",
    "stratified": "stratified_kfold",
    "stratified_k_fold": "stratified_kfold",
    "stratifiedkfold": "stratified_kfold",
    "loo": "leave_one_out",
    "loocv": "leave_one_out",
    "leaveoneout": "leave_one_out",
    "timeseries": "time_series",
    "time_series_split": "time_series",
    "chronological": "time_series",
    "groupkfold": "group_kfold",
    "group_k_fold": "group_kfold",
    "logo": "leave_one_group_out",
    "leaveonegroupout": "leave_one_group_out",
    "repeatedkfold": "repeated_kfold",
    "repeated_k_fold": "repeated_kfold",
    "repeatedstratifiedkfold": "repeated_stratified_kfold",
    "shufflesplit": "shuffle_split",
    "monte_carlo": "shuffle_split",
    "stratifiedshufflesplit": "stratified_shuffle_split",
    "walkforward": "walk_forward",
    "rolling_window": "walk_forward",
    "rolling": "walk_forward",
    "oob": "bootstrap",
    "out_of_bag": "bootstrap",
}
_SHUFFLED_CV = frozenset({"kfold", "stratified_kfold"})
_RANDOM_CV = frozenset({
    "repeated_kfold", "repeated_stratified_kfold", "shuffle_split", "stratified_shuffle_split", "bootstrap",
})
_FOLDED_CV = frozenset({
    "kfold", "stratified_kfold", "time_series", "group_kfold", "repeated_kfold",
    "repeated_stratified_kfold", "shuffle_split", "stratified_shuffle_split", "walk_forward",
})
_STRATIFIED_CV = frozenset({"stratified_kfold", "repeated_stratified_kfold", "stratified_shuffle_split"})
_GROUPED_CV = frozenset({"group_kfold", "leave_one_group_out"})
_REPEATED_CV = frozenset({"repeated_kfold", "repeated_stratified_kfold"})
_SHUFFLE_SPLIT_CV = frozenset({"shuffle_split", "stratified_shuffle_split"})

# ---------------------------------------------------------------------------
# Input schemas (strict) — source of truth for LLM_TOOL_SCHEMAS
# ---------------------------------------------------------------------------


class SoftSensorAlgorithm(str, Enum):
    OLS = "OLS"  # plain linear regression (ordinary least squares)
    PLS = "PLS"
    RIDGE = "RIDGE"
    LASSO = "LASSO"
    PCR = "PCR"
    KNN = "KNN"


_ALGORITHM_ALIASES = {
    "LINEAR": "OLS",
    "LINEAR_REGRESSION": "OLS",
    "LINEARREGRESSION": "OLS",
    "LR": "OLS",
    "MLR": "OLS",
    "K_NN": "KNN",
}


class TrainSoftSensorArgs(BaseModel):
    """Arguments for training an OLS or PLS soft-sensor regression model."""

    dataset_version_id: str = Field(
        ...,
        description="UUID of the dataset version used for training/testing.",
        min_length=36,
        max_length=36,
    )
    algorithm: SoftSensorAlgorithm = Field(
        ...,
        description=(
            "Regression algorithm: OLS (plain linear regression), PLS, RIDGE, "
            "LASSO, PCR, or KNN."
        ),
    )
    target_column: str = Field(
        ...,
        description="Measured quality / soft-sensor target column (y).",
        min_length=1,
    )
    feature_columns: list[str] = Field(
        ...,
        description="Process variable / sensor feature columns (X).",
        min_length=1,
    )
    n_components: int = Field(
        default=2,
        description="Number of latent components for PLS/PCR.",
        ge=1,
    )
    alpha: float = Field(
        default=1.0,
        description="Regularization strength for Ridge/Lasso.",
        ge=0,
    )
    n_neighbors: int = Field(
        default=5,
        description="Number of neighbors for k-NN.",
        ge=1,
    )
    split_strategy: str = Field(
        default="random",
        description=(
            "How to split data into train/test: "
            "'random' (shuffled train_test_split), "
            "'stratified' (shuffled, keeping each stratum's share equal in train and test; strata "
            "are stratify_column, or the target binned into n_strata quantile bins), "
            "'sequential' (chronological / time-series split: the last test_size fraction of rows "
            "is the test set, preserving time order), or 'group' (GroupShuffleSplit: whole groups "
            "of group_column go to train or test, test_size is the share of groups)."
        ),
        json_schema_extra={"enum": list(_SPLIT_STRATEGIES)},
    )
    test_size: float = Field(
        default=_TEST_SIZE,
        description="Fraction of rows held out as the test set (0.2 = 80/20 train/test split).",
        ge=0.05,
        le=0.5,
    )
    validation_size: float = Field(
        default=0.0,
        description=(
            "Fraction of rows held out as a separate validation set (holdout validation with "
            "train, validation AND test sets), carved from the training part with the same "
            "split_strategy; 0 for none. test_size + validation_size can be at most 0.8."
        ),
        ge=0,
        le=0.4,
    )
    group_column: Optional[str] = Field(
        default=None,
        description=(
            "Column of group ids (batch, plant, patient …) for the group split and for "
            "group_kfold / leave_one_group_out, so one group never appears in both training and "
            "evaluation rows."
        ),
    )
    random_state: int = Field(
        default=_RANDOM_STATE,
        description="Random seed of the shuffled splits (random and stratified split, shuffled K-Fold).",
    )
    stratify_column: Optional[str] = Field(
        default=None,
        description=(
            "Column whose groups the stratified split and stratified K-Fold keep in proportion. "
            "Omit to stratify on the target. A numeric column with more than n_strata distinct "
            "values is binned into n_strata quantile bins."
        ),
    )
    n_strata: int = Field(
        default=5,
        description="Quantile bins used to stratify a continuous target or column.",
        ge=2,
        le=20,
    )
    cv_strategy: str = Field(
        default="none",
        description=(
            "Cross-validation on the training rows, reported next to the test-set scores: "
            "'none', 'kfold' (K-Fold), 'stratified_kfold' (Stratified K-Fold on the same strata "
            "as the stratified split), 'leave_one_out' (LOOCV: one fit per training row, at most "
            f"{_MAX_LOO_ROWS} rows), 'time_series' (expanding-window folds in row order), "
            "'group_kfold' (GroupKFold on group_column), 'leave_one_group_out' (one fold per "
            "group), 'repeated_kfold' / 'repeated_stratified_kfold' (K-Fold repeated cv_repeats "
            "times with new shuffles), 'shuffle_split' / 'stratified_shuffle_split' (cv_folds "
            "random train/validation splits of cv_test_size), 'walk_forward' (strict time-series "
            "validation: a rolling training window of cv_window rows, each followed by the next "
            "block), or 'bootstrap' (n_bootstrap resamples with replacement, scored on the "
            "out-of-bag rows)."
        ),
        json_schema_extra={"enum": list(_CV_STRATEGIES)},
    )
    cv_folds: int = Field(
        default=5,
        description=(
            "Number of folds (kfold, stratified_kfold, time_series, group_kfold, repeated, "
            "walk_forward) or of random splits (shuffle_split, stratified_shuffle_split)."
        ),
        ge=2,
        le=20,
    )
    cv_shuffle: bool = Field(
        default=True,
        description="Shuffle the rows (with random_state) before cutting K-Fold / Stratified K-Fold folds.",
    )
    cv_repeats: int = Field(
        default=3,
        description="For repeated_kfold and repeated_stratified_kfold: how many times K-Fold is repeated.",
        ge=2,
        le=10,
    )
    cv_test_size: float = Field(
        default=0.2,
        description="For shuffle_split and stratified_shuffle_split: share of the training rows in each validation split.",
        ge=0.05,
        le=0.5,
    )
    cv_window: Optional[int] = Field(
        default=None,
        description=(
            "For walk_forward: rows in each rolling training window. Omit to use the size of the "
            "first time-series fold."
        ),
        ge=2,
    )
    n_bootstrap: int = Field(
        default=50,
        description="For bootstrap: number of resamples (one fit each).",
        ge=10,
        le=500,
    )
    fit_intercept: bool = Field(
        default=True,
        description="Fit an intercept term (OLS, Ridge, Lasso).",
    )
    pls_scale: bool = Field(
        default=True,
        description="Standardize X and y inside PLS.",
    )
    max_iter: int = Field(
        default=1000,
        description="Maximum solver iterations for Lasso.",
        ge=100,
        le=100000,
    )
    knn_weights: str = Field(
        default="uniform",
        description="k-NN neighbour weighting: 'uniform' or 'distance'.",
        json_schema_extra={"enum": ["uniform", "distance"]},
    )

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        return _validate_uuid_str(value)

    @field_validator("algorithm", mode="before")
    @classmethod
    def _algorithm_alias(cls, v: Any) -> Any:
        if isinstance(v, str):
            key = v.strip().upper().replace("-", "_").replace(" ", "_")
            return _ALGORITHM_ALIASES.get(key, key)
        return v

    @field_validator("split_strategy")
    @classmethod
    def _valid_split_strategy(cls, v: str) -> str:
        key = v.strip().lower().replace("-", "_").replace(" ", "_")
        key = _SPLIT_ALIASES.get(key, key)
        if key not in _SPLIT_STRATEGIES:
            raise ValueError(f"split_strategy must be one of {list(_SPLIT_STRATEGIES)}")
        return key

    @field_validator("cv_strategy", mode="before")
    @classmethod
    def _valid_cv_strategy(cls, v: Any) -> str:
        key = "none" if v is None else str(v).strip().lower().replace("-", "_").replace(" ", "_")
        key = _CV_ALIASES.get(key, key)
        if key not in _CV_STRATEGIES:
            raise ValueError(f"cv_strategy must be one of {list(_CV_STRATEGIES)}")
        return key

    @field_validator("stratify_column", "group_column", mode="before")
    @classmethod
    def _blank_stratify(cls, v: Any) -> Any:
        return v.strip() or None if isinstance(v, str) else v

    @model_validator(mode="after")
    def _consistent(self) -> "TrainSoftSensorArgs":
        if self.test_size + self.validation_size > 0.8 + 1e-9:
            raise ValueError("test_size + validation_size can be at most 0.8, so some rows are left to train on")
        needs_groups = self.split_strategy == "group" or self.cv_strategy in _GROUPED_CV
        if needs_groups and not self.group_column:
            raise ValueError(f"{'the group split' if self.split_strategy == 'group' else self.cv_strategy} needs group_column")
        if self.group_column and self.group_column == self.target_column:
            raise ValueError("group_column cannot be the target")
        return self

    @field_validator("knn_weights")
    @classmethod
    def _valid_knn_weights(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in {"uniform", "distance"}:
            raise ValueError("knn_weights must be 'uniform' or 'distance'")
        return v

    @field_validator("feature_columns")
    @classmethod
    def _non_empty_features(cls, value: list[str]) -> list[str]:
        cleaned = [c.strip() for c in value if c and c.strip()]
        if not cleaned:
            raise ValueError("feature_columns must contain at least one column")
        leaked = [c for c in cleaned if c.startswith("Predicted_")]
        if leaked:
            raise ValueError(
                f"{leaked} hold an earlier model's predictions; using them as features "
                "leaks the target. Remove them from feature_columns."
            )
        return cleaned

    @field_validator("target_column")
    @classmethod
    def _strip_target(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("target_column must not be empty")
        return cleaned


_MODEL_ID_DESCRIPTION = (
    "Optional UUID of a trained model. Omit it to use the most recently trained "
    "model on the active dataset version or any version it was derived from."
)


class GenerateDiagnosticPlotArgs(BaseModel):
    """Shared arguments for parity, residuals, importance, and coefficient plots."""

    dataset_version_id: str = Field(
        ...,
        description="UUID of the dataset version to score with the saved model.",
        min_length=36,
        max_length=36,
    )
    model_id: Optional[str] = Field(default=None, description=_MODEL_ID_DESCRIPTION)

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        return _validate_uuid_str(value)

    @field_validator("model_id")
    @classmethod
    def _lenient_model_id(cls, value: Optional[str]) -> Optional[str]:
        # A malformed id falls back to lineage lookup instead of failing the call.
        try:
            return _validate_uuid_str(value) if value else None
        except ValueError:
            return None


GenerateParityPlotArgs = GenerateDiagnosticPlotArgs


class GenerateClassificationCurveArgs(GenerateDiagnosticPlotArgs):
    """ROC and precision-recall curves of a regression soft sensor used as an alarm."""

    threshold: Optional[float] = Field(
        default=None,
        description=(
            "Target value that defines the event, e.g. a quality limit: rows above it (see "
            "positive_when) are positives and the prediction is the score. Omit for a two-valued "
            "target (its larger value is the event) or the median of the target."
        ),
    )
    positive_when: str = Field(
        default="above",
        description="'above': values above threshold are the event; 'below': values below it are.",
        json_schema_extra={"enum": ["above", "below"]},
    )

    @field_validator("positive_when")
    @classmethod
    def _direction(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in {"above", "below"}:
            raise ValueError("positive_when must be 'above' or 'below'")
        return value


class LearningCurveMetric(str, Enum):
    rmse = "rmse"
    r2 = "r2"


class GenerateLearningCurveArgs(GenerateDiagnosticPlotArgs):
    """Learning curve of a trained model's settings refit on growing training sets."""

    cv_folds: int = Field(default=5, description="Cross-validation folds at each training size.", ge=2, le=10)
    n_points: int = Field(default=6, description="Number of training sizes, from small to all of the model's training rows.", ge=3, le=10)
    metric: LearningCurveMetric = Field(default=LearningCurveMetric.rmse, description="Score plotted: rmse or r2.")
    random_state: int = Field(default=_RANDOM_STATE, description="Seed of the shuffled folds.")


class ListTrainedModelsArgs(BaseModel):
    """Arguments for listing models trained on the active dataset lineage."""

    dataset_version_id: str = Field(..., description="UUID of the dataset version.", min_length=36, max_length=36)

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        return _validate_uuid_str(value)


# ---------------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------------


def build_model_path(owner_id: str, dataset_id: uuid.UUID | str, model_id: uuid.UUID | str) -> str:
    """Storage key for a soft-sensor model artifact."""
    return model_key(owner_id, dataset_id, model_id)


def _save_model_artifact(key: str, artifact: dict[str, Any]) -> str:
    get_storage().write(key, lambda path: joblib.dump(artifact, path))
    return key


def _load_model_artifact(key: str) -> dict[str, Any]:
    # Artifacts are only ever written by train_soft_sensor under the owner's key.
    artifact = joblib.load(get_storage().fetch(key))
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError(f"Invalid soft-sensor artifact: {key}")
    return artifact


def _prepare_xy(
    df: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    if target_column in feature_columns:
        raise ValueError("target_column must not appear in feature_columns")
    _assert_numeric_columns(df, [*feature_columns, target_column])
    subset = df[feature_columns + [target_column]].dropna()
    if len(subset) < 5:
        raise ValueError(
            f"Need at least 5 complete rows after dropping NA; found {len(subset)}"
        )
    return subset[feature_columns], subset[target_column]


def _fit_estimator(
    algorithm: SoftSensorAlgorithm,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_components: int,
    alpha: float = 1.0,
    n_neighbors: int = 5,
    *,
    fit_intercept: bool = True,
    pls_scale: bool = True,
    max_iter: int = 1000,
    knn_weights: str = "uniform",
) -> Any:
    if algorithm == SoftSensorAlgorithm.OLS:
        model = LinearRegression(fit_intercept=fit_intercept)
        model.fit(X_train, y_train)
        return model
    elif algorithm == SoftSensorAlgorithm.RIDGE:
        model = Ridge(alpha=alpha, fit_intercept=fit_intercept)
        model.fit(X_train, y_train)
        return model
    elif algorithm == SoftSensorAlgorithm.LASSO:
        model = Lasso(alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter)
        model.fit(X_train, y_train)
        return model
    elif algorithm == SoftSensorAlgorithm.KNN:
        if n_neighbors > len(X_train):
            raise ValueError(
                f"n_neighbors={n_neighbors} exceeds the {len(X_train)} training rows"
            )
        model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=knn_weights)
        model.fit(X_train, y_train)
        return model

    max_components = min(X_train.shape[0], X_train.shape[1])
    if n_components > max_components:
        raise ValueError(
            f"n_components={n_components} exceeds max allowed for this split "
            f"({max_components})"
        )
    
    if algorithm == SoftSensorAlgorithm.PCR:
        model = make_pipeline(PCA(n_components=n_components), LinearRegression())
        model.fit(X_train, y_train)
        return model
    else:
        # Default PLS
        model = PLSRegression(n_components=n_components, scale=pls_scale)
        model.fit(X_train, y_train)
        return model


def _extract_coefficients(
    model: Any,
    *,
    algorithm: SoftSensorAlgorithm,
    feature_columns: list[str],
) -> dict[str, float]:
    """Return per-feature regression coefficients / importances."""
    if algorithm == SoftSensorAlgorithm.KNN:
        return {} # Non-parametric, no coefficients
        
    if algorithm == SoftSensorAlgorithm.PCR:
        pca = model.named_steps['pca']
        ols = model.named_steps['linearregression']
        raw = np.dot(pca.components_.T, ols.coef_).reshape(-1)
    else:
        raw = np.asarray(model.coef_, dtype=float).reshape(-1)
        
    if raw.size != len(feature_columns):
        # PLS may expose (n_targets, n_features); flatten safely.
        raw = raw.reshape(-1)[: len(feature_columns)]
    coeffs = {
        name: float(value)
        for name, value in zip(feature_columns, raw, strict=False)
    }
    
    if algorithm in (SoftSensorAlgorithm.OLS, SoftSensorAlgorithm.RIDGE, SoftSensorAlgorithm.LASSO) and hasattr(model, "intercept_"):
        coeffs["intercept"] = float(np.asarray(model.intercept_).reshape(-1)[0])
    elif algorithm == SoftSensorAlgorithm.PCR:
        coeffs["intercept"] = float(np.asarray(model.named_steps['linearregression'].intercept_).reshape(-1)[0])
        
    return coeffs


def _predict(model: Any, X: pd.DataFrame) -> np.ndarray:
    y_hat = model.predict(X)
    return np.asarray(y_hat, dtype=float).reshape(-1)


def _strata(source: pd.Series, n_strata: int) -> pd.Series:
    """Group labels for stratifying: a continuous series is cut into equal-count quantile bins."""
    if (
        pd.api.types.is_numeric_dtype(source)
        and not pd.api.types.is_bool_dtype(source)
        and source.nunique() > n_strata
    ):
        labels = pd.qcut(source.rank(method="first"), q=n_strata, labels=False)
    else:
        labels = source
    return labels.astype(object).where(labels.notna(), "missing").astype(str)


def _group_labels(source: pd.Series) -> pd.Series:
    return source.astype(object).where(source.notna(), "missing").astype(str)


def _split(
    X: pd.DataFrame,
    y: pd.Series,
    args: TrainSoftSensorArgs,
    strata: Optional[pd.Series] = None,
    groups: Optional[pd.Series] = None,
    *,
    held_out: Optional[float] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split off ``held_out`` (a fraction of these rows, default test_size) with split_strategy."""
    fraction = args.test_size if held_out is None else held_out
    if args.split_strategy == "sequential":
        # Time-order preserving: the last `fraction` of rows is held out.
        n_test = max(1, int(len(X) * fraction))
        n_train = len(X) - n_test
        if n_train < 2:
            raise ValueError(f"The sequential split leaves {n_train} training row(s); lower test_size / validation_size")
        return X.iloc[:n_train], X.iloc[n_train:], y.iloc[:n_train], y.iloc[n_train:]
    if args.split_strategy == "group":
        labels = groups if groups is not None else None
        if labels is None:
            raise ValueError("The group split needs group_column")
        if labels.nunique() < 2:
            raise ValueError(f"The group split needs at least 2 groups in {args.group_column!r}")
        splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=args.random_state)
        try:
            train_idx, test_idx = next(splitter.split(X, y, labels))
        except ValueError as exc:
            raise ValueError(f"The group split failed: {exc} Use more groups or a larger test_size.") from exc
        return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
    stratify = None
    if args.split_strategy == "stratified":
        stratify = strata if strata is not None else _strata(y, args.n_strata)
        smallest = int(stratify.value_counts().min())
        if smallest < 2:
            raise ValueError(
                f"A stratified split needs at least 2 rows in every stratum; the smallest of "
                f"{stratify.nunique()} strata has {smallest}. Lower n_strata or pick another stratify_column."
            )
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=fraction, random_state=args.random_state, stratify=stratify
        )
    except ValueError as exc:
        if stratify is None:
            raise
        raise ValueError(f"The stratified split failed: {exc} Lower n_strata or raise test_size.") from exc
    return X_train, X_test, y_train, y_test


def _holdout(
    X: pd.DataFrame,
    y: pd.Series,
    args: TrainSoftSensorArgs,
    strata: Optional[pd.Series],
    groups: Optional[pd.Series],
) -> tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame, pd.Series, Optional[pd.Series], pd.Series]:
    """Train, validation (when validation_size > 0) and test rows."""
    X_rest, X_test, y_rest, y_test = _split(X, y, args, strata, groups)
    if args.validation_size <= 0:
        return X_rest, None, X_test, y_rest, None, y_test
    share = args.validation_size / (1 - args.test_size)
    X_train, X_val, y_train, y_val = _split(
        X_rest, y_rest, args,
        strata.loc[X_rest.index] if strata is not None else None,
        groups.loc[X_rest.index] if groups is not None else None,
        held_out=share,
    )
    if len(X_val) < 2:
        raise ValueError(f"The validation set has {len(X_val)} row(s); raise validation_size")
    return X_train, X_val, X_test, y_train, y_val, y_test


def _fit(args: TrainSoftSensorArgs, X: pd.DataFrame, y: pd.Series) -> Any:
    return _fit_estimator(
        args.algorithm,
        X,
        y,
        args.n_components,
        args.alpha,
        args.n_neighbors,
        fit_intercept=args.fit_intercept,
        pls_scale=args.pls_scale,
        max_iter=args.max_iter,
        knn_weights=args.knn_weights,
    )


def _finite(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _check_strata(labels: pd.Series, k: int, what: str) -> None:
    smallest = int(labels.value_counts().min())
    if smallest < k:
        raise ValueError(
            f"{what} needs at least {k} training rows in every stratum; "
            f"the smallest has {smallest}. Lower cv_folds or n_strata."
        )


def _bootstrap_folds(n: int, rounds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    folds = []
    for _ in range(rounds):
        drawn = rng.integers(0, n, n)
        out_of_bag = np.setdiff1d(np.arange(n), drawn)
        if len(out_of_bag):
            folds.append((drawn, out_of_bag))
    return folds


def _cv_folds(
    args: TrainSoftSensorArgs, X: pd.DataFrame, strata: Optional[pd.Series], groups: Optional[pd.Series] = None
) -> list[tuple[np.ndarray, np.ndarray]]:
    n = len(X)
    strategy = args.cv_strategy
    if strategy == "leave_one_out":
        if n > _MAX_LOO_ROWS:
            raise ValueError(
                f"Leave-one-out would fit {n} models; it is limited to {_MAX_LOO_ROWS} training rows. "
                "Use kfold instead."
            )
        return list(LeaveOneOut().split(X))
    if strategy == "bootstrap":
        return _bootstrap_folds(n, args.n_bootstrap, args.random_state)
    if strategy in _GROUPED_CV:
        if groups is None:
            raise ValueError(f"{strategy} needs group_column")
        n_groups = int(groups.nunique())
        if strategy == "leave_one_group_out":
            if n_groups < 2:
                raise ValueError(f"leave_one_group_out needs at least 2 groups in {args.group_column!r}")
            if n_groups > _MAX_CV_FITS:
                raise ValueError(f"leave_one_group_out would fit {n_groups} models (at most {_MAX_CV_FITS}); use group_kfold")
            return list(LeaveOneGroupOut().split(X, groups=groups))
        if args.cv_folds > n_groups:
            raise ValueError(f"group_kfold with {args.cv_folds} folds needs at least {args.cv_folds} groups; {args.group_column!r} has {n_groups}")
        return list(GroupKFold(n_splits=args.cv_folds).split(X, groups=groups))
    k = args.cv_folds
    if strategy in ("time_series", "walk_forward"):
        if k >= n:
            raise ValueError(f"cv_folds={k} needs more than {k} training rows; there are {n}")
        if strategy == "time_series":
            return list(TimeSeriesSplit(n_splits=k).split(X))
        window = args.cv_window or max(2, n // (k + 1))
        if window >= n:
            raise ValueError(f"cv_window={window} must be smaller than the {n} training rows")
        return list(TimeSeriesSplit(n_splits=k, max_train_size=window).split(X))
    if strategy in _SHUFFLE_SPLIT_CV:
        if strategy == "stratified_shuffle_split":
            _check_strata(strata, 2, "Stratified shuffle split")
            splitter = StratifiedShuffleSplit(n_splits=k, test_size=args.cv_test_size, random_state=args.random_state)
            try:
                return list(splitter.split(X, strata))
            except ValueError as exc:
                raise ValueError(f"The stratified shuffle split failed: {exc} Lower n_strata or raise cv_test_size.") from exc
        return list(ShuffleSplit(n_splits=k, test_size=args.cv_test_size, random_state=args.random_state).split(X))
    if k > n:
        raise ValueError(f"cv_folds={k} is more than the {n} training rows")
    if strategy == "repeated_kfold":
        return list(RepeatedKFold(n_splits=k, n_repeats=args.cv_repeats, random_state=args.random_state).split(X))
    if strategy == "repeated_stratified_kfold":
        _check_strata(strata, k, f"Repeated Stratified K-Fold with {k} folds")
        return list(
            RepeatedStratifiedKFold(n_splits=k, n_repeats=args.cv_repeats, random_state=args.random_state).split(X, strata)
        )
    seed = args.random_state if args.cv_shuffle else None
    if strategy == "stratified_kfold":
        _check_strata(strata, k, f"Stratified K-Fold with {k} folds")
        return list(StratifiedKFold(n_splits=k, shuffle=args.cv_shuffle, random_state=seed).split(X, strata))
    return list(KFold(n_splits=k, shuffle=args.cv_shuffle, random_state=seed).split(X))


def _cross_validate(
    args: TrainSoftSensorArgs,
    X: pd.DataFrame,
    y: pd.Series,
    strata: Optional[pd.Series],
    groups: Optional[pd.Series] = None,
) -> Optional[dict[str, Any]]:
    """Scores on the training rows: pooled out-of-fold R² / RMSE plus per-fold spread.

    Rows evaluated in several splits (repeated K-Fold, shuffle splits,
    bootstrap) contribute the mean of their predictions to the pooled score.
    """
    if args.cv_strategy == "none":
        return None
    if args.cv_strategy in _STRATIFIED_CV and strata is None:
        strata = _strata(y, args.n_strata)
    folds = _cv_folds(args, X, strata, groups)
    if not folds:
        raise ValueError(f"{args.cv_strategy} produced no validation rows")
    y_all = y.to_numpy(dtype=float)
    total = np.zeros(len(X))
    count = np.zeros(len(X))
    per_fold: list[dict[str, Any]] = []
    for i, (train_idx, val_idx) in enumerate(folds, start=1):
        model = _fit(args, X.iloc[train_idx], y.iloc[train_idx])
        pred = _predict(model, X.iloc[val_idx])
        np.add.at(total, val_idx, pred)
        np.add.at(count, val_idx, 1)
        true = y_all[val_idx]
        per_fold.append({
            "fold": i,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "r2": _finite(r2_score(true, pred)) if len(val_idx) >= 2 else None,
            "rmse": _finite(np.sqrt(mean_squared_error(true, pred))),
        })
    scored = count > 0
    pooled = np.divide(total, count, out=np.full(len(X), np.nan), where=scored)
    fold_r2 = [f["r2"] for f in per_fold if f["r2"] is not None]
    fold_rmse = [f["rmse"] for f in per_fold if f["rmse"] is not None]
    return {
        "strategy": args.cv_strategy,
        "n_splits": len(per_fold),
        "n_rows": int(scored.sum()),
        "r2": _finite(r2_score(y_all[scored], pooled[scored])) if scored.sum() >= 2 else None,
        "rmse": _finite(np.sqrt(mean_squared_error(y_all[scored], pooled[scored]))),
        "fold_r2_mean": _finite(np.mean(fold_r2)) if fold_r2 else None,
        "fold_r2_std": _finite(np.std(fold_r2)) if fold_r2 else None,
        "fold_rmse_mean": _finite(np.mean(fold_rmse)) if fold_rmse else None,
        "fold_rmse_std": _finite(np.std(fold_rmse)) if fold_rmse else None,
        # Leave-one-out has one fold per row; listing them would swamp the UI and the chat.
        "folds": per_fold if len(per_fold) <= 20 else None,
    }


_ALGORITHM_HYPERPARAMETERS: dict[SoftSensorAlgorithm, tuple[str, ...]] = {
    SoftSensorAlgorithm.OLS: ("fit_intercept",),
    SoftSensorAlgorithm.RIDGE: ("alpha", "fit_intercept"),
    SoftSensorAlgorithm.LASSO: ("alpha", "fit_intercept", "max_iter"),
    SoftSensorAlgorithm.PLS: ("n_components", "pls_scale"),
    SoftSensorAlgorithm.PCR: ("n_components",),
    SoftSensorAlgorithm.KNN: ("n_neighbors", "knn_weights"),
}


def _hyperparameters(args: TrainSoftSensorArgs) -> dict[str, Any]:
    """The settings that shaped this model, for lineage and the model registry."""
    out: dict[str, Any] = {"split_strategy": args.split_strategy, "test_size": args.test_size}
    if args.validation_size > 0:
        out["validation_size"] = args.validation_size
    shuffled_cv = (args.cv_strategy in _SHUFFLED_CV and args.cv_shuffle) or args.cv_strategy in _RANDOM_CV
    if args.split_strategy in ("random", "stratified", "group") or shuffled_cv:
        out["random_state"] = args.random_state
    if args.split_strategy == "stratified" or args.cv_strategy in _STRATIFIED_CV:
        out["stratify_column"] = args.stratify_column or args.target_column
        out["n_strata"] = args.n_strata
    if args.split_strategy == "group" or args.cv_strategy in _GROUPED_CV:
        out["group_column"] = args.group_column
    if args.cv_strategy != "none":
        out["cv_strategy"] = args.cv_strategy
        if args.cv_strategy in _FOLDED_CV:
            out["cv_folds"] = args.cv_folds
        if args.cv_strategy in _SHUFFLED_CV:
            out["cv_shuffle"] = args.cv_shuffle
        if args.cv_strategy in _REPEATED_CV:
            out["cv_repeats"] = args.cv_repeats
        if args.cv_strategy in _SHUFFLE_SPLIT_CV:
            out["cv_test_size"] = args.cv_test_size
        if args.cv_strategy == "walk_forward" and args.cv_window is not None:
            out["cv_window"] = args.cv_window
        if args.cv_strategy == "bootstrap":
            out["n_bootstrap"] = args.n_bootstrap
    for name in _ALGORITHM_HYPERPARAMETERS[args.algorithm]:
        out[name] = getattr(args, name)
    return out


def _training_rows(
    df: pd.DataFrame, args: TrainSoftSensorArgs
) -> tuple[pd.DataFrame, pd.Series, Optional[pd.Series], Optional[pd.Series]]:
    """Complete rows (X, y) with the strata and group labels the split and CV need."""
    X, y = _prepare_xy(df, feature_columns=args.feature_columns, target_column=args.target_column)
    strata = None
    if args.split_strategy == "stratified" or args.cv_strategy in _STRATIFIED_CV:
        if args.stratify_column and args.stratify_column not in df.columns:
            raise ValueError(f"stratify_column {args.stratify_column!r} is not a column of this dataset")
        source = df.loc[X.index, args.stratify_column] if args.stratify_column else y
        strata = _strata(source, args.n_strata)
    groups = None
    if args.split_strategy == "group" or args.cv_strategy in _GROUPED_CV:
        if args.group_column not in df.columns:
            raise ValueError(f"group_column {args.group_column!r} is not a column of this dataset")
        groups = _group_labels(df.loc[X.index, args.group_column])
    return X, y, strata, groups


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


async def train_soft_sensor(
    dataset_version_id: str,
    algorithm: str,
    target_column: str,
    feature_columns: list[str],
    n_components: int = 2,
    alpha: float = 1.0,
    n_neighbors: int = 5,
    split_strategy: str = "random",
    test_size: float = _TEST_SIZE,
    random_state: int = _RANDOM_STATE,
    fit_intercept: bool = True,
    pls_scale: bool = True,
    max_iter: int = 1000,
    knn_weights: str = "uniform",
    stratify_column: Optional[str] = None,
    n_strata: int = 5,
    cv_strategy: str = "none",
    cv_folds: int = 5,
    cv_shuffle: bool = True,
    validation_size: float = 0.0,
    group_column: Optional[str] = None,
    cv_repeats: int = 3,
    cv_test_size: float = 0.2,
    cv_window: Optional[int] = None,
    n_bootstrap: int = 50,
    *,
    session: Optional[AsyncSession] = None,
) -> dict[str, Any]:
    """
    Train a soft sensor on a train/test split (``test_size`` held out, 0.2 by default),
    optionally with a validation set and cross-validated on the training rows.

    Persists the ``.pkl`` artifact through storage and returns metrics plus
    feature coefficients for process interpretation.
    """
    args = TrainSoftSensorArgs(
        dataset_version_id=dataset_version_id,
        algorithm=algorithm,  # type: ignore[arg-type]
        target_column=target_column,
        feature_columns=feature_columns,
        n_components=n_components,
        alpha=alpha,
        n_neighbors=n_neighbors,
        split_strategy=split_strategy,
        test_size=test_size,
        random_state=random_state,
        fit_intercept=fit_intercept,
        pls_scale=pls_scale,
        max_iter=max_iter,
        knn_weights=knn_weights,
        stratify_column=stratify_column,
        n_strata=n_strata,
        cv_strategy=cv_strategy,
        cv_folds=cv_folds,
        cv_shuffle=cv_shuffle,
        validation_size=validation_size,
        group_column=group_column,
        cv_repeats=cv_repeats,
        cv_test_size=cv_test_size,
        cv_window=cv_window,
        n_bootstrap=n_bootstrap,
    )

    async def _run(db: AsyncSession) -> dict[str, Any]:
        version = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(version)
        X, y, strata, groups = _training_rows(df, args)
        X_train, X_val, X_test, y_train, y_val, y_test = _holdout(X, y, args, strata, groups)
        cv = _cross_validate(
            args, X_train, y_train,
            strata.loc[X_train.index] if strata is not None else None,
            groups.loc[X_train.index] if groups is not None else None,
        )
        model = _fit(args, X_train, y_train)
        hyperparameters = _hyperparameters(args)
        y_pred = _predict(model, X_test)
        r2 = float(r2_score(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        train_pred = _predict(model, X_train)
        train = {
            "n_rows": int(len(X_train)),
            "r2": _finite(r2_score(y_train, train_pred)),
            "rmse": _finite(np.sqrt(mean_squared_error(y_train, train_pred))),
        }
        validation = None
        if X_val is not None:
            val_pred = _predict(model, X_val)
            validation = {
                "n_rows": int(len(X_val)),
                "r2": _finite(r2_score(y_val, val_pred)),
                "rmse": _finite(np.sqrt(mean_squared_error(y_val, val_pred))),
            }
        coefficients = _extract_coefficients(
            model,
            algorithm=args.algorithm,
            feature_columns=args.feature_columns,
        )

        model_id = uuid.uuid4()
        
        # Make predictions on the full dataset and save a new dataset version
        # X excludes rows with NA inputs (e.g. the first rows of lag features); they stay NaN.
        full_pred = pd.Series(_predict(model, X), index=X.index)
        df[f"Predicted_{args.target_column}"] = full_pred.reindex(df.index)
        
        new_version_id, _ = await _persist_new_version(
            db,
            parent=version,
            df=df,
            action_performed=f"trained_model_for_{args.target_column}",
            parameters_used={
                "model_id": str(model_id),
                "algorithm": args.algorithm.value,
                "target": args.target_column,
                "features": args.feature_columns,
                **hyperparameters,
            },
            summary_message=(
                f"Trained {args.algorithm.value} soft sensor for "
                f"{args.target_column} and stored predictions."
            ),
        )

        artifact = {
            "model_id": str(model_id),
            "algorithm": args.algorithm.value,
            "model": model,
            "feature_columns": args.feature_columns,
            "target_column": args.target_column,
            "n_components": args.n_components,
            "hyperparameters": hyperparameters,
            "dataset_version_id": args.dataset_version_id,
            "new_version_id": new_version_id,
            "metrics": {"r2_score": r2, "rmse": rmse},
            "cv": cv,
            "train": train,
            "validation": validation,
            "coefficients": coefficients,
            "train_size": int(len(X_train)),
            "train_index": X_train.index.tolist(),
            "test_index": X_test.index.tolist(),
            "validation_rows": int(len(X_val)) if X_val is not None else 0,
            "test_size": int(len(X_test)),
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        dataset = await db.get(Dataset, version.dataset_id)
        if dataset is None:
            raise ValueError(f"Dataset not found: {version.dataset_id}")
        artifact_path = _save_model_artifact(
            build_model_path(dataset.owner_id, dataset.id, model_id), artifact
        )
        await register_model(db, dataset_id=dataset.id, artifact=artifact, artifact_key=artifact_path)

        return {
            "model_id": str(model_id),
            "algorithm": args.algorithm.value,
            "new_version_id": new_version_id,
            "r2_score": r2,
            "rmse": rmse,
            "feature_importances": coefficients,
            "coefficients": coefficients,
            "artifact_path": artifact_path,
            "split_strategy": args.split_strategy,
            "test_size": args.test_size,
            "hyperparameters": hyperparameters,
            "cv": cv,
            "train": train,
            "validation": validation,
            "n_train": int(len(X_train)),
            "n_validation": int(len(X_val)) if X_val is not None else 0,
            "n_test": int(len(X_test)),
            "target_column": args.target_column,
            "feature_columns": args.feature_columns,
        }

    if session is not None:
        return await _run(session)
    async with AsyncSessionLocal() as db:
        return await _run(db)


async def generate_parity_plot(
    dataset_version_id: str,
    model_id: Optional[str] = None,
    *,
    session: Optional[AsyncSession] = None,
) -> dict[str, Any]:
    """
    Score a saved soft sensor and return a Plotly Actual vs Predicted parity plot.

    Includes a red 45-degree reference line ($y = x$) for visual bias/variance checks.
    """
    args = GenerateParityPlotArgs(
        dataset_version_id=dataset_version_id,
        model_id=model_id,
    )

    async def _run(db: AsyncSession) -> dict[str, Any]:
        artifact = _load_model_artifact(
            (await resolve_model(db, args.dataset_version_id, args.model_id)).artifact_key
        )
        target_column: str = str(artifact["target_column"])
        _X_train, _y_train, X, y = await _model_split(db, artifact)
        y_hat = _predict(artifact["model"], X)
        y_true = y.to_numpy(dtype=float)
        scores = _test_scores(y_true, y_hat)

        lo = float(min(y_true.min(), y_hat.min()))
        hi = float(max(y_true.max(), y_hat.max()))
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        axis_min, axis_max = lo - pad, hi + pad

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=y_true,
                y=y_hat,
                mode="markers",
                name="Test rows",
                marker={"size": 8, "opacity": 0.75},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[axis_min, axis_max],
                y=[axis_min, axis_max],
                mode="lines",
                name="y = x",
                line={"color": "red", "width": 2, "dash": "solid"},
            )
        )
        fig.update_layout(
            title=(
                f"Parity plot — {artifact.get('algorithm', 'model')} (target: {target_column}) · "
                f"{len(y_true)} held-out test rows · {scores}"
            ),
            xaxis_title="Actual (y)",
            yaxis_title="Predicted (ŷ)",
            xaxis={"range": [axis_min, axis_max], "zeroline": False},
            yaxis={
                "range": [axis_min, axis_max],
                "scaleanchor": "x",
                "scaleratio": 1,
                "zeroline": False,
            },
            template="plotly_white",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        )
        return json.loads(fig.to_json())

    if session is not None:
        return await _run(session)
    async with AsyncSessionLocal() as db:
        return await _run(db)


# ---------------------------------------------------------------------------
# Diagnostic plot tools (Residuals, Variable Importance, Coefficients)
# ---------------------------------------------------------------------------


async def generate_residuals_plot(
    dataset_version_id: str,
    model_id: Optional[str] = None,
    *,
    session: Optional[AsyncSession] = None,
) -> dict[str, Any]:
    """
    Generate a residuals diagnostic plot for a trained soft-sensor model.

    Shows two panels: (1) Residuals vs Predicted values — to detect
    heteroscedasticity or systematic bias, and (2) Residuals vs Index —
    to detect trends or drift over time.
    """
    args = GenerateDiagnosticPlotArgs(
        dataset_version_id=dataset_version_id,
        model_id=model_id,
    )

    async def _run(db: AsyncSession) -> dict[str, Any]:
        artifact = _load_model_artifact(
            (await resolve_model(db, args.dataset_version_id, args.model_id)).artifact_key
        )
        target_column: str = str(artifact["target_column"])
        _X_train, _y_train, X, y = await _model_split(db, artifact)
        X, y = X.sort_index(), y.sort_index()
        y_hat = _predict(artifact["model"], X)
        residuals = np.asarray(y.to_numpy(dtype=float)) - y_hat
        positions = [int(p) for p in X.index] if pd.api.types.is_integer_dtype(X.index) else list(range(len(X)))

        from plotly.subplots import make_subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Residuals vs Predicted", "Residuals vs Index"),
        )
        # Panel 1: Residuals vs Predicted
        fig.add_trace(
            go.Scatter(
                x=y_hat.tolist(), y=residuals.tolist(),
                mode="markers",
                name="Residuals",
                marker={"size": 6, "opacity": 0.6, "color": "steelblue"},
            ),
            row=1, col=1,
        )
        # Zero-line
        fig.add_hline(y=0, line_dash="dot", line_color="red", row=1, col=1)

        # Panel 2: Residuals vs observation index (time drift)
        fig.add_trace(
            go.Scatter(
                x=positions, y=residuals.tolist(),
                mode="markers+lines",
                name="Residuals (index)",
                marker={"size": 4, "opacity": 0.5, "color": "steelblue"},
                line={"width": 1},
            ),
            row=1, col=2,
        )
        fig.add_hline(y=0, line_dash="dot", line_color="red", row=1, col=2)

        fig.update_layout(
            title=(
                f"Residuals diagnostic — {artifact.get('algorithm', 'model')} (target: {target_column}) · "
                f"{len(residuals)} held-out test rows"
            ),
            template="plotly_white",
            showlegend=False,
        )
        fig.update_xaxes(title_text="Predicted (ŷ)", row=1, col=1)
        fig.update_yaxes(title_text="Residuals (y − ŷ)", row=1, col=1)
        fig.update_xaxes(title_text="Row number in the dataset", row=1, col=2)
        fig.update_yaxes(title_text="Residuals (y − ŷ)", row=1, col=2)

        return json.loads(fig.to_json())

    if session is not None:
        return await _run(session)
    async with AsyncSessionLocal() as db:
        return await _run(db)


async def generate_importance_plot(
    dataset_version_id: str,
    model_id: Optional[str] = None,
    *,
    session: Optional[AsyncSession] = None,
) -> dict[str, Any]:
    """
    Generate a Variable Importance / VIP bar chart for a trained soft-sensor model.

    Shows the absolute magnitude of each feature's regression coefficient
    (or PLS VIP score for PLS), ranked from most to least important.
    """
    args = GenerateDiagnosticPlotArgs(
        dataset_version_id=dataset_version_id,
        model_id=model_id,
    )

    async def _run(db: AsyncSession) -> dict[str, Any]:
        artifact = _load_model_artifact(
            (await resolve_model(db, args.dataset_version_id, args.model_id)).artifact_key
        )
        coefficients: dict[str, float] = artifact.get("coefficients", {})
        algorithm = artifact.get("algorithm", "model")
        target_column = artifact.get("target_column", "target")

        # Filter out intercept
        feat_coeff = {k: abs(v) for k, v in coefficients.items() if k != "intercept"}
        if not feat_coeff:
            raise ValueError("No feature coefficients found in this model artifact.")

        # Sort descending by absolute magnitude
        sorted_items = sorted(feat_coeff.items(), key=lambda kv: kv[1], reverse=True)
        features, importances = zip(*sorted_items)

        # Color bars by sign (original, before abs)
        original_signs = [coefficients.get(f, 0) for f in features]
        bar_colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in original_signs]

        fig = go.Figure(go.Bar(
            x=list(importances),
            y=list(features),
            orientation="h",
            marker_color=bar_colors,
            text=[f"{v:.4f}" for v in importances],
            textposition="outside",
        ))
        fig.update_layout(
            title=f"Feature importance — {algorithm} (target: {target_column})",
            xaxis_title="|Coefficient| / Importance",
            yaxis_title="Feature",
            template="plotly_white",
            yaxis={"autorange": "reversed"},
            height=max(300, 40 * len(features)),
        )
        return json.loads(fig.to_json())

    if session is not None:
        return await _run(session)
    async with AsyncSessionLocal() as db:
        return await _run(db)


async def generate_coefficients_plot(
    dataset_version_id: str,
    model_id: Optional[str] = None,
    *,
    session: Optional[AsyncSession] = None,
) -> dict[str, Any]:
    """
    Generate a signed coefficients bar chart for a trained soft-sensor model.

    Unlike the importance plot, this preserves the sign of each coefficient,
    clearly showing which features have a positive vs negative impact on the target.
    """
    args = GenerateDiagnosticPlotArgs(
        dataset_version_id=dataset_version_id,
        model_id=model_id,
    )

    async def _run(db: AsyncSession) -> dict[str, Any]:
        artifact = _load_model_artifact(
            (await resolve_model(db, args.dataset_version_id, args.model_id)).artifact_key
        )
        coefficients: dict[str, float] = artifact.get("coefficients", {})
        algorithm = artifact.get("algorithm", "model")
        target_column = artifact.get("target_column", "target")

        # Filter out intercept; keep signs
        feat_coeff = {k: v for k, v in coefficients.items() if k != "intercept"}
        if not feat_coeff:
            raise ValueError("No feature coefficients found in this model artifact.")

        sorted_items = sorted(feat_coeff.items(), key=lambda kv: kv[1])
        features, coeff_values = zip(*sorted_items)
        bar_colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in coeff_values]

        fig = go.Figure(go.Bar(
            x=list(coeff_values),
            y=list(features),
            orientation="h",
            marker_color=bar_colors,
            text=[f"{v:+.4f}" for v in coeff_values],
            textposition="outside",
        ))
        fig.add_vline(x=0, line_dash="dot", line_color="gray")
        fig.update_layout(
            title=f"Regression coefficients — {algorithm} (target: {target_column})",
            xaxis_title="Coefficient value",
            yaxis_title="Feature",
            template="plotly_white",
            height=max(300, 40 * len(features)),
        )
        return json.loads(fig.to_json())

    if session is not None:
        return await _run(session)
    async with AsyncSessionLocal() as db:
        return await _run(db)


def _artifact_args(artifact: dict[str, Any]) -> TrainSoftSensorArgs:
    """The training arguments of a saved model, from its recorded hyperparameters."""
    recorded = dict(artifact.get("hyperparameters") or {})
    fixed = {"dataset_version_id", "algorithm", "target_column", "feature_columns"}
    allowed = set(TrainSoftSensorArgs.model_fields) - fixed
    return TrainSoftSensorArgs(
        dataset_version_id=str(artifact["dataset_version_id"]),
        algorithm=artifact["algorithm"],
        target_column=str(artifact["target_column"]),
        feature_columns=list(artifact["feature_columns"]),
        **{k: v for k, v in recorded.items() if k in allowed},
    )


async def _model_split(
    db: AsyncSession, artifact: dict[str, Any]
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """(X_train, y_train, X_test, y_test) of the dataset version the model was trained on."""
    try:
        train_args = _artifact_args(artifact)
        trained_on = _load_dataframe(await _get_version(db, train_args.dataset_version_id))
        X, y, strata, groups = _training_rows(trained_on, train_args)
        if "test_index" in artifact and "train_index" in artifact:
            train_index, test_index = artifact["train_index"], artifact["test_index"]
            return X.loc[train_index], y.loc[train_index], X.loc[test_index], y.loc[test_index]
        X_train, _X_val, X_test, y_train, _y_val, y_test = _holdout(X, y, train_args, strata, groups)
        return X_train, y_train, X_test, y_test
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Could not recover this model's train/test split ({exc}); retrain the model") from exc


def _test_scores(y_true: np.ndarray, y_hat: np.ndarray) -> str:
    if len(y_true) < 2:
        return "too few rows for R²"
    rmse = float(np.sqrt(mean_squared_error(y_true, y_hat)))
    return f"R² {r2_score(y_true, y_hat):.3f}, RMSE {rmse:.3g}"


def _events(y: np.ndarray, scores: np.ndarray, args: GenerateClassificationCurveArgs) -> tuple[np.ndarray, np.ndarray, str]:
    """Binary truth and scores for the curve, and a description of the event."""
    values = np.unique(y)
    if args.threshold is None and len(values) == 2:
        truth = (y == values[1]).astype(int)
        return truth, scores, f"target = {values[1]:g}"
    threshold = float(np.median(y)) if args.threshold is None else args.threshold
    if args.positive_when == "above":
        truth, described = (y > threshold).astype(int), f"target > {threshold:g}"
    else:
        truth, described, scores = (y < threshold).astype(int), f"target < {threshold:g}", -scores
    return truth, scores, described


async def _classification_curve(
    args: GenerateClassificationCurveArgs, session: Optional[AsyncSession], kind: str
) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

    async def _run(db: AsyncSession) -> dict[str, Any]:
        artifact = _load_model_artifact(
            (await resolve_model(db, args.dataset_version_id, args.model_id)).artifact_key
        )
        _X_train, _y_train, X, y = await _model_split(db, artifact)
        rows = f"{len(X)} held-out test rows"
        y_true = y.to_numpy(dtype=float)
        truth, scores, event = _events(y_true, _predict(artifact["model"], X), args)
        positives = int(truth.sum())
        if positives == 0 or positives == len(truth):
            raise ValueError(f"Every evaluated row is {'an event' if positives else 'a non-event'} ({event}); choose another threshold")
        algorithm = artifact.get("algorithm", "model")
        fig = go.Figure()
        if kind == "roc":
            fpr, tpr, _ = roc_curve(truth, scores)
            auc = float(roc_auc_score(truth, scores))
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC = {auc:.3f})", line={"width": 2}))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="chance", line={"dash": "dash", "color": "gray"}))
            fig.update_layout(
                title=f"ROC curve — {algorithm}, event: {event} ({rows})",
                xaxis_title="False positive rate", yaxis_title="True positive rate",
            )
        else:
            precision, recall, _ = precision_recall_curve(truth, scores)
            ap = float(average_precision_score(truth, scores))
            prevalence = positives / len(truth)
            fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=f"PR (AP = {ap:.3f})", line={"width": 2, "shape": "hv"}))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[prevalence, prevalence], mode="lines", name=f"baseline ({prevalence:.2f})",
                line={"dash": "dash", "color": "gray"},
            ))
            fig.update_layout(
                title=f"Precision-recall curve — {algorithm}, event: {event} ({rows})",
                xaxis_title="Recall", yaxis_title="Precision",
            )
        fig.update_layout(
            template="plotly_white", xaxis_range=[0, 1], yaxis_range=[0, 1.02],
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        )
        return json.loads(fig.to_json())

    if session is not None:
        return await _run(session)
    async with AsyncSessionLocal() as db:
        return await _run(db)


async def generate_roc_curve(
    dataset_version_id: str,
    model_id: Optional[str] = None,
    threshold: Optional[float] = None,
    positive_when: str = "above",
    *,
    session: Optional[AsyncSession] = None,
) -> dict[str, Any]:
    """ROC curve, on the model's held-out test rows, of the soft sensor used to flag a threshold crossing."""
    args = GenerateClassificationCurveArgs(
        dataset_version_id=dataset_version_id, model_id=model_id, threshold=threshold,
        positive_when=positive_when,
    )
    return await _classification_curve(args, session, "roc")


async def generate_pr_curve(
    dataset_version_id: str,
    model_id: Optional[str] = None,
    threshold: Optional[float] = None,
    positive_when: str = "above",
    *,
    session: Optional[AsyncSession] = None,
) -> dict[str, Any]:
    """Precision-recall curve, on the model's held-out test rows, of the soft sensor used to flag a threshold crossing."""
    args = GenerateClassificationCurveArgs(
        dataset_version_id=dataset_version_id, model_id=model_id, threshold=threshold,
        positive_when=positive_when,
    )
    return await _classification_curve(args, session, "pr")


def _smallest_training_size(artifact: dict[str, Any]) -> int:
    hyper = artifact.get("hyperparameters") or {}
    algorithm = str(artifact.get("algorithm"))
    if algorithm == SoftSensorAlgorithm.KNN.value:
        return int(hyper.get("n_neighbors", 5)) + 1
    if algorithm in (SoftSensorAlgorithm.PLS.value, SoftSensorAlgorithm.PCR.value):
        return int(hyper.get("n_components", artifact.get("n_components", 2))) + 1
    return 3


async def generate_learning_curve(
    dataset_version_id: str,
    model_id: Optional[str] = None,
    cv_folds: int = 5,
    n_points: int = 6,
    metric: str = "rmse",
    random_state: int = _RANDOM_STATE,
    *,
    session: Optional[AsyncSession] = None,
) -> dict[str, Any]:
    """Training and cross-validated scores of the model's settings as the training set grows."""
    from sklearn.base import clone
    from sklearn.model_selection import learning_curve

    args = GenerateLearningCurveArgs(
        dataset_version_id=dataset_version_id, model_id=model_id, cv_folds=cv_folds,
        n_points=n_points, metric=metric, random_state=random_state,  # type: ignore[arg-type]
    )

    async def _run(db: AsyncSession) -> dict[str, Any]:
        artifact = _load_model_artifact(
            (await resolve_model(db, args.dataset_version_id, args.model_id)).artifact_key
        )
        X, y, _X_test, _y_test = await _model_split(db, artifact)
        largest = len(X) - math.ceil(len(X) / args.cv_folds)
        smallest = max(_smallest_training_size(artifact), int(0.1 * largest))
        if largest < smallest:
            raise ValueError(f"Too few training rows ({len(X)}) for a learning curve with {args.cv_folds} folds")
        sizes = np.unique(np.linspace(smallest, largest, args.n_points).astype(int))
        scoring = "neg_root_mean_squared_error" if args.metric == LearningCurveMetric.rmse else "r2"
        sizes, train_scores, val_scores = learning_curve(
            clone(artifact["model"]), X, y, train_sizes=sizes,
            cv=KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state),
            scoring=scoring, error_score="raise", shuffle=True, random_state=args.random_state,
        )
        sign = -1.0 if args.metric == LearningCurveMetric.rmse else 1.0
        label = "RMSE" if args.metric == LearningCurveMetric.rmse else "R²"
        fig = go.Figure()
        for name, scores, color in (("Training", train_scores, "#636efa"), ("Cross-validation", val_scores, "#ef553b")):
            mean = sign * scores.mean(axis=1)
            std = scores.std(axis=1)
            fig.add_trace(go.Scatter(
                x=np.concatenate([sizes, sizes[::-1]]), y=np.concatenate([mean + std, (mean - std)[::-1]]),
                fill="toself", fillcolor=color, opacity=0.15, line={"width": 0}, hoverinfo="skip", showlegend=False,
            ))
            fig.add_trace(go.Scatter(x=sizes, y=mean, mode="lines+markers", name=name, line={"color": color}))
        fig.update_layout(
            title=(
                f"Learning curve — {artifact.get('algorithm', 'model')} (target: {artifact['target_column']}, "
                f"{args.cv_folds}-fold CV on the {len(X)} training rows; test rows unused)"
            ),
            xaxis_title="Training rows", yaxis_title=f"{label} (mean ± std over folds)",
            template="plotly_white", legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        )
        return json.loads(fig.to_json())

    if session is not None:
        return await _run(session)
    async with AsyncSessionLocal() as db:
        return await _run(db)


async def list_trained_models(
    dataset_version_id: str,
    *,
    session: Optional[AsyncSession] = None,
) -> dict[str, Any]:
    """List models trained on this dataset version or its ancestors, newest first."""
    args = ListTrainedModelsArgs(dataset_version_id=dataset_version_id)

    async def _run(db: AsyncSession) -> dict[str, Any]:
        models = await lineage_models(db, args.dataset_version_id)
        return {"count": len(models), "models": models}

    if session is not None:
        return await _run(session)
    async with AsyncSessionLocal() as db:
        return await _run(db)


# ---------------------------------------------------------------------------
# OpenAI / Groq tool schema export + dispatch map
# ---------------------------------------------------------------------------

LLM_TOOL_SCHEMAS: list[dict[str, Any]] = [
    _openai_tool_schema(
        name="train_soft_sensor",
        description=(
            "Train a soft-sensor regression model (linear regression / OLS, PLS, "
            "Ridge, Lasso, PCR or k-NN) on a dataset version. test_size sets the held-out "
            "fraction (default 0.2, i.e. an 80/20 train/test split); validation_size adds a "
            "separate validation set (holdout train/validation/test). split_strategy picks a "
            "random, stratified, sequential (chronological / time-series) or group split. cv_strategy "
            "adds cross-validation on the training rows: kfold, stratified_kfold, leave_one_out "
            "(LOOCV), time_series, group_kfold, leave_one_group_out, repeated_kfold, "
            "repeated_stratified_kfold, shuffle_split, stratified_shuffle_split, walk_forward or "
            "bootstrap (out-of-bag). "
            "Reports test-set R² and RMSE, saves a .pkl artifact under "
            "data_storage/models/, and returns model_id, metrics, and feature "
            "coefficients/importances."
        ),
        args_model=TrainSoftSensorArgs,
    ),
    _openai_tool_schema(
        name="generate_parity_plot",
        description=(
            "Load a trained soft-sensor .pkl model, predict its held-out test rows only "
            "(never the training rows) and return a Plotly parity scatter plot "
            "(Actual on X, Predicted on Y) with a red y = x reference line and the test R² "
            "and RMSE in the title."
        ),
        args_model=GenerateParityPlotArgs,
    ),
    _openai_tool_schema(
        name="generate_residuals_plot",
        description=(
            "Generate a two-panel residuals diagnostic plot of a trained model on its held-out "
            "test rows only: (1) Residuals vs Predicted — to detect heteroscedasticity or bias, "
            "(2) Residuals vs row number — to detect time drift. "
            "Returns Plotly JSON for UI rendering."
        ),
        args_model=GenerateDiagnosticPlotArgs,
    ),
    _openai_tool_schema(
        name="generate_importance_plot",
        description=(
            "Generate a horizontal bar chart of feature importances (absolute coefficient magnitudes) "
            "for a trained model, ranked from most to least important. "
            "Green bars = positive effect, red bars = negative effect. "
            "Returns Plotly JSON for UI rendering."
        ),
        args_model=GenerateDiagnosticPlotArgs,
    ),
    _openai_tool_schema(
        name="generate_coefficients_plot",
        description=(
            "Generate a signed regression coefficients bar chart for a trained model, "
            "clearly showing which features push the target up (positive, green) "
            "vs down (negative, red). Returns Plotly JSON for UI rendering."
        ),
        args_model=GenerateDiagnosticPlotArgs,
    ),
    _openai_tool_schema(
        name="generate_roc_curve",
        description=(
            "ROC curve (with AUC) of a trained soft sensor used as an alarm: rows whose target "
            "is above (or below) threshold are events and the prediction is the score. Uses the "
            "model's held-out test rows only. Returns Plotly JSON."
        ),
        args_model=GenerateClassificationCurveArgs,
    ),
    _openai_tool_schema(
        name="generate_pr_curve",
        description=(
            "Precision-recall curve (with average precision) of a trained soft sensor used as an "
            "alarm for target values above (or below) threshold, on the model's held-out test rows "
            "only. Better than ROC for rare events. Returns Plotly JSON."
        ),
        args_model=GenerateClassificationCurveArgs,
    ),
    _openai_tool_schema(
        name="generate_learning_curve",
        description=(
            "Learning curve of a trained model's algorithm and settings: training and "
            "cross-validated RMSE (or R²) as the number of training rows grows, using only the "
            "model's training rows (its test rows stay unseen), to diagnose under- vs over-fitting "
            "and whether more data would help. Returns Plotly JSON."
        ),
        args_model=GenerateLearningCurveArgs,
    ),
    _openai_tool_schema(
        name="list_trained_models",
        description=(
            "List the soft-sensor models already trained on the active dataset version "
            "or any version it was derived from, newest first, with algorithm, target, "
            "features, R² and RMSE. Use it to find or compare existing models."
        ),
        args_model=ListTrainedModelsArgs,
    ),
]

TOOL_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "list_trained_models": list_trained_models,
    "train_soft_sensor": train_soft_sensor,
    "generate_parity_plot": generate_parity_plot,
    "generate_residuals_plot": generate_residuals_plot,
    "generate_importance_plot": generate_importance_plot,
    "generate_coefficients_plot": generate_coefficients_plot,
    "generate_roc_curve": generate_roc_curve,
    "generate_pr_curve": generate_pr_curve,
    "generate_learning_curve": generate_learning_curve,
}

__all__ = [
    "GenerateClassificationCurveArgs",
    "GenerateLearningCurveArgs",
    "GenerateParityPlotArgs",
    "GenerateDiagnosticPlotArgs",
    "ListTrainedModelsArgs",
    "list_trained_models",
    "LLM_TOOL_SCHEMAS",
    "SoftSensorAlgorithm",
    "TOOL_FUNCTIONS",
    "TrainSoftSensorArgs",
    "build_model_path",
    "generate_parity_plot",
    "generate_residuals_plot",
    "generate_importance_plot",
    "generate_coefficients_plot",
    "generate_learning_curve",
    "generate_pr_curve",
    "generate_roc_curve",
    "train_soft_sensor",
]
