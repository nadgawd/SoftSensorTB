"""Live checks on a deployed soft sensor.

* Input health: a missing reading, a value outside the training range, or a
  sensor stuck on one value.
* Operating region: Hotelling's T² (how far the inputs are from the training
  operating point, inside the main PCA directions) and SPE / Q (how much the
  inputs break the correlations seen in training). Limits are the 99th
  percentile of the training rows, so about 1% of normal rows exceed them.
* Bias update from lab results: corrected = prediction + bias, and every lab
  value moves the bias toward the latest error by a gain g:
  bias ← (1 − g)·bias + g·(lab − prediction).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

# A sensor repeating one value this many times while it varied in training is reported as stuck.
STUCK_RUN = 20
_STUCK_MIN_UNIQUE = 10
# Values a little outside the training range are normal; beyond this share of the range they are reported.
RANGE_MARGIN = 0.05
_PCA_VARIANCE = 0.90
_PCA_MAX_COMPONENTS = 10
_LIMIT_QUANTILE = 0.99
ACCURACY_WINDOW = 20
ACCURACY_MIN_LABS = 5
ACCURACY_ALARM_RATIO = 2.0


@dataclass
class InputStats:
    numeric: bool
    low: Optional[float] = None
    high: Optional[float] = None
    unique: int = 0
    categories: list[str] = field(default_factory=list)
    typical: Optional[float] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "numeric": self.numeric, "low": self.low, "high": self.high, "typical": self.typical,
            "unique": self.unique, "categories": self.categories[:50],
        }


def input_stats(cut_data: pd.DataFrame, inputs: list[str], time_column: Optional[str]) -> dict[str, InputStats]:
    """The range and variety of every live input in the data the live feed corresponds to."""
    out: dict[str, InputStats] = {}
    for col in inputs:
        if col == time_column:
            continue
        values = cut_data[col]
        if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
            finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            out[col] = InputStats(
                numeric=True,
                low=float(finite.min()) if len(finite) else None,
                high=float(finite.max()) if len(finite) else None,
                unique=int(finite.nunique()),
                typical=float(finite.median()) if len(finite) else None,
            )
        else:
            seen = values.dropna().astype(str)
            out[col] = InputStats(numeric=False, unique=int(seen.nunique()), categories=sorted(seen.unique().tolist()))
    return out


def input_findings(
    frame: pd.DataFrame,
    stats: dict[str, InputStats],
    new_mask: np.ndarray,
) -> tuple[list[list[str]], list[list[str]]]:
    """For each new row: (missing inputs, warnings about the inputs it has).

    ``frame`` holds the recent history followed by the new rows (typed), so a
    stuck sensor is seen across requests.
    """
    rows = int(new_mask.sum())
    missing: list[list[str]] = [[] for _ in range(rows)]
    warnings: list[list[str]] = [[] for _ in range(rows)]
    positions = np.flatnonzero(new_mask)
    for col, st in stats.items():
        if col not in frame.columns:
            for i in range(rows):
                missing[i].append(col)
            continue
        values = frame[col]
        absent = values.isna().to_numpy()
        if st.numeric:
            numbers = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
            span = (st.high - st.low) if st.low is not None and st.high is not None else None
            margin = RANGE_MARGIN * span if span else 0.0
            stuck_run = _run_lengths(numbers) if st.unique > _STUCK_MIN_UNIQUE else None
        for i, pos in enumerate(positions):
            if absent[pos]:
                missing[i].append(col)
                continue
            if st.numeric:
                x = numbers[pos]
                if not math.isfinite(x):
                    missing[i].append(col)
                    continue
                if st.low is not None and x < st.low - margin:
                    warnings[i].append(f"{col} = {_fmt(x)} is below the training range (min {_fmt(st.low)})")
                elif st.high is not None and x > st.high + margin:
                    warnings[i].append(f"{col} = {_fmt(x)} is above the training range (max {_fmt(st.high)})")
                if stuck_run is not None and stuck_run[pos] >= STUCK_RUN:
                    warnings[i].append(f"{col} has read {_fmt(x)} for {int(stuck_run[pos])} rows in a row (stuck sensor?)")
            elif st.categories and str(values.iloc[pos]) not in st.categories:
                warnings[i].append(f"{col} = {values.iloc[pos]!r} never appeared in training")
    return missing, warnings


def _run_lengths(x: np.ndarray) -> np.ndarray:
    """At each position, how many values in a row (ending there) are equal."""
    out = np.ones(len(x))
    for i in range(1, len(x)):
        if math.isfinite(x[i]) and x[i] == x[i - 1]:
            out[i] = out[i - 1] + 1
    return out


def _fmt(x: float) -> str:
    return f"{x:.4g}"


@dataclass
class RegionModel:
    """PCA of the standardised training features, with T² and SPE limits."""

    features: list[str]
    mean: np.ndarray
    scale: np.ndarray
    loadings: np.ndarray  # features × k
    variances: np.ndarray  # k
    t2_limit: float
    spe_limit: Optional[float]

    def scores(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Z = (X - self.mean) / self.scale
        T = Z @ self.loadings
        t2 = np.sum(T**2 / self.variances, axis=1)
        residual = Z - T @ self.loadings.T
        spe = np.sum(residual**2, axis=1)
        return t2, spe

    def as_dict(self) -> dict[str, Any]:
        return {
            "components": int(self.loadings.shape[1]),
            "t2_limit": self.t2_limit,
            "spe_limit": self.spe_limit,
        }


def fit_region(X: pd.DataFrame) -> Optional[RegionModel]:
    """None when there are too few complete rows to describe the operating region."""
    data = X.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    n, p = data.shape
    if n < max(20, p + 2):
        return None
    mean = data.mean().to_numpy()
    scale = data.std(ddof=1).to_numpy()
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    Z = (data.to_numpy() - mean) / scale
    _, s, vt = np.linalg.svd(Z, full_matrices=False)
    variances = s**2 / (n - 1)
    total = variances.sum()
    if total <= 0:
        return None
    share = np.cumsum(variances) / total
    k = int(np.searchsorted(share, _PCA_VARIANCE) + 1)
    k = max(1, min(k, _PCA_MAX_COMPONENTS, p, n - 1))
    keep = variances[:k] > 1e-12
    k = max(1, int(keep.sum()))
    loadings = vt[:k].T
    model = RegionModel(
        features=list(X.columns), mean=mean, scale=scale, loadings=loadings,
        variances=variances[:k], t2_limit=0.0, spe_limit=None,
    )
    t2, spe = model.scores(data.to_numpy())
    model.t2_limit = float(np.quantile(t2, _LIMIT_QUANTILE))
    # With every direction kept the residual is only rounding noise, so SPE says nothing.
    model.spe_limit = float(np.quantile(spe, _LIMIT_QUANTILE)) if k < p and np.quantile(spe, _LIMIT_QUANTILE) > 1e-10 else None
    return model


def region_findings(region: Optional[RegionModel], X: np.ndarray) -> tuple[list[Optional[float]], list[Optional[float]], list[list[str]]]:
    if region is None or len(X) == 0:
        return [None] * len(X), [None] * len(X), [[] for _ in range(len(X))]
    t2, spe = region.scores(X)
    notes: list[list[str]] = []
    for a, b in zip(t2, spe):
        row: list[str] = []
        if a > region.t2_limit:
            row.append(f"Inputs are far from the training operating point (T² {a:.3g} > limit {region.t2_limit:.3g})")
        if region.spe_limit is not None and b > region.spe_limit:
            row.append(f"Inputs break the relationships seen in training (SPE {b:.3g} > limit {region.spe_limit:.3g})")
        notes.append(row)
    return [float(v) for v in t2], [float(v) for v in spe], notes


def update_bias(bias: float, gain: float, residual: float) -> float:
    return (1.0 - gain) * bias + gain * residual


def accuracy(residuals: list[float], test_rmse: Optional[float]) -> dict[str, Any]:
    """Recent lab accuracy of the corrected prediction, against the test-set error."""
    recent = [r for r in residuals[-ACCURACY_WINDOW:] if r is not None and math.isfinite(r)]
    if not recent:
        return {"n": 0, "rmse": None, "bias": None, "test_rmse": test_rmse, "alarm": False, "message": ""}
    rmse = float(np.sqrt(np.mean(np.square(recent))))
    mean_error = float(np.mean(recent))
    alarm = bool(
        test_rmse and len(recent) >= ACCURACY_MIN_LABS and rmse > ACCURACY_ALARM_RATIO * test_rmse
    )
    message = (
        f"Lab error over the last {len(recent)} samples (RMSE {rmse:.4g}) is more than "
        f"{ACCURACY_ALARM_RATIO:g}× the test error ({test_rmse:.4g}); the process may have changed — consider retraining."
        if alarm else ""
    )
    return {"n": len(recent), "rmse": rmse, "bias": mean_error, "test_rmse": test_rmse, "alarm": alarm, "message": message}
