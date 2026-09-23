"""Soft-sensor training and validation tools for Groq OpenAI-compatible calling.

OLS / PLS regression models are fit on dataset versions, persisted under the
storage key ``{owner}/{dataset}/models/{model_id}.pkl``, and validated with parity plots for
process soft-sensor performance assessment.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pydantic import BaseModel, Field, field_validator
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
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

# ---------------------------------------------------------------------------
# Input schemas (strict) — source of truth for LLM_TOOL_SCHEMAS
# ---------------------------------------------------------------------------


class SoftSensorAlgorithm(str, Enum):
    OLS = "OLS"
    PLS = "PLS"
    RIDGE = "RIDGE"
    LASSO = "LASSO"
    PCR = "PCR"
    KNN = "KNN"


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
        description="Regression algorithm: OLS, PLS, RIDGE, LASSO, PCR, or KNN.",
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
            "'random' (sklearn train_test_split) or "
            "'sequential' (last 20% of rows as test, preserving time order)."
        ),
    )

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        return _validate_uuid_str(value)

    @field_validator("split_strategy")
    @classmethod
    def _valid_split_strategy(cls, v: str) -> str:
        if v not in {"random", "sequential"}:
            raise ValueError("split_strategy must be 'random' or 'sequential'")
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
) -> Any:
    if algorithm == SoftSensorAlgorithm.OLS:
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model
    elif algorithm == SoftSensorAlgorithm.RIDGE:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        return model
    elif algorithm == SoftSensorAlgorithm.LASSO:
        model = Lasso(alpha=alpha)
        model.fit(X_train, y_train)
        return model
    elif algorithm == SoftSensorAlgorithm.KNN:
        model = KNeighborsRegressor(n_neighbors=n_neighbors)
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
        model = PLSRegression(n_components=n_components, scale=True)
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
    *,
    session: Optional[AsyncSession] = None,
) -> dict[str, Any]:
    """
    Train a soft sensor on an 80/20 train/test split.

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
    )

    async def _run(db: AsyncSession) -> dict[str, Any]:
        version = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(version)
        X, y = _prepare_xy(
            df,
            feature_columns=args.feature_columns,
            target_column=args.target_column,
        )

        if args.split_strategy == "sequential":
            # Time-series safe split: last 20% of rows are test set
            n_test = max(1, int(len(X) * _TEST_SIZE))
            n_train = len(X) - n_test
            X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
            y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=_TEST_SIZE,
                random_state=_RANDOM_STATE,
            )
        model = _fit_estimator(
            args.algorithm,
            X_train,
            y_train,
            args.n_components,
            args.alpha,
            args.n_neighbors,
        )
        y_pred = _predict(model, X_test)
        r2 = float(r2_score(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
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
                "split_strategy": args.split_strategy,
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
            "dataset_version_id": args.dataset_version_id,
            "new_version_id": new_version_id,
            "metrics": {"r2_score": r2, "rmse": rmse},
            "coefficients": coefficients,
            "train_size": int(len(X_train)),
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
            "n_train": int(len(X_train)),
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
        version = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(version)
        artifact = _load_model_artifact(
            (await resolve_model(db, args.dataset_version_id, args.model_id)).artifact_key
        )

        feature_columns: list[str] = list(artifact["feature_columns"])
        target_column: str = str(artifact["target_column"])
        model = artifact["model"]

        X, y = _prepare_xy(
            df,
            feature_columns=feature_columns,
            target_column=target_column,
        )
        y_hat = _predict(model, X)
        y_true = y.to_numpy(dtype=float)

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
                name="Soft sensor",
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
                f"Parity plot — {artifact.get('algorithm', 'model')} "
                f"(target: {target_column})"
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
        version = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(version)
        artifact = _load_model_artifact(
            (await resolve_model(db, args.dataset_version_id, args.model_id)).artifact_key
        )

        feature_columns: list[str] = list(artifact["feature_columns"])
        target_column: str = str(artifact["target_column"])
        model = artifact["model"]

        X, y = _prepare_xy(df, feature_columns=feature_columns, target_column=target_column)
        y_hat = _predict(model, X)
        residuals = np.asarray(y.to_numpy(dtype=float)) - y_hat

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
                x=list(range(len(residuals))), y=residuals.tolist(),
                mode="markers+lines",
                name="Residuals (index)",
                marker={"size": 4, "opacity": 0.5, "color": "steelblue"},
                line={"width": 1},
            ),
            row=1, col=2,
        )
        fig.add_hline(y=0, line_dash="dot", line_color="red", row=1, col=2)

        fig.update_layout(
            title=f"Residuals diagnostic — {artifact.get('algorithm', 'model')} (target: {target_column})",
            template="plotly_white",
            showlegend=False,
        )
        fig.update_xaxes(title_text="Predicted (ŷ)", row=1, col=1)
        fig.update_yaxes(title_text="Residuals (y − ŷ)", row=1, col=1)
        fig.update_xaxes(title_text="Observation index", row=1, col=2)
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
            "Train a soft-sensor regression model (OLS, PLS, Ridge, Lasso, PCR or "
            "k-NN) on a dataset version with an 80/20 train/test split. "
            "Reports test-set R² and RMSE, saves a .pkl artifact under "
            "data_storage/models/, and returns model_id, metrics, and feature "
            "coefficients/importances."
        ),
        args_model=TrainSoftSensorArgs,
    ),
    _openai_tool_schema(
        name="generate_parity_plot",
        description=(
            "Load a trained soft-sensor .pkl model and a dataset version, compute "
            "predicted vs actual values, and return a Plotly parity scatter plot "
            "(Actual on X, Predicted on Y) with a red y = x reference line for "
            "UI rendering."
        ),
        args_model=GenerateParityPlotArgs,
    ),
    _openai_tool_schema(
        name="generate_residuals_plot",
        description=(
            "Generate a two-panel residuals diagnostic plot for a trained model: "
            "(1) Residuals vs Predicted — to detect heteroscedasticity or bias, "
            "(2) Residuals vs Index — to detect time drift. "
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
}

__all__ = [
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
    "train_soft_sensor",
]
