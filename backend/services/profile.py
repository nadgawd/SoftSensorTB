"""Data quality profile of a whole dataset version, for the Data quality step and the summary bar.

Plant historian data has its own failure modes, so besides missing values this
looks for stuck (flat-lined) sensors, outliers, numbers stored as text, the
sampling interval and gaps of the time column, and how often the (usually
sparse) lab target was measured. A sparse target is expected and is reported as
coverage, not as a problem.
"""

from __future__ import annotations

from collections import OrderedDict
from threading import Lock
from typing import Any, Optional

import numpy as np
import pandas as pd

# A run of at least this many identical consecutive readings counts as flat-lined.
STUCK_RUN = 20
# Share of rows inside such runs above which a sensor is flagged as possibly stuck.
STUCK_SHARE_WARNING = 0.2
# Robust z-score (median / MAD) beyond which a reading counts as an outlier.
OUTLIER_Z = 6.0
# A step this many times the median sampling interval counts as a gap.
GAP_FACTOR = 5.0
MISSING_WARNING = 0.05
MISSING_PROBLEM = 0.5
_TIME_NAME_HINTS = ("time", "date", "timestamp", "datetime")

_cache: "OrderedDict[tuple, dict[str, Any]]" = OrderedDict()
_cache_lock = Lock()
_CACHE_SIZE = 16


def cached_profile(key: tuple, df_loader, **kwargs) -> dict[str, Any]:
    """Versions are immutable, so a profile is computed once per (version, options)."""
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    result = profile_dataframe(df_loader(), **kwargs)
    with _cache_lock:
        _cache[key] = result
        while len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)
    return result


def _num(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _iso(value: Any) -> Optional[str]:
    return None if value is None or pd.isna(value) else pd.Timestamp(value).isoformat()


def _as_datetime(series: pd.Series) -> Optional[pd.Series]:
    """The series as datetimes when it is (or reliably parses as) a time column."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return None
    sample = series.dropna().head(200)
    if sample.empty:
        return None
    parsed = pd.to_datetime(sample, errors="coerce", utc=True, format="mixed")
    if parsed.notna().mean() < 0.9:
        return None
    return pd.to_datetime(series, errors="coerce", utc=True, format="mixed")


def detect_time_column(df: pd.DataFrame, preferred: Optional[str] = None) -> Optional[str]:
    if preferred and preferred in df.columns and _as_datetime(df[preferred]) is not None:
        return preferred
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    for col in df.columns:
        if any(h in str(col).lower() for h in _TIME_NAME_HINTS) and _as_datetime(df[col]) is not None:
            return col
    return None


def _time_profile(times: pd.Series) -> dict[str, Any]:
    valid = times.dropna()
    info: dict[str, Any] = {
        "start": _iso(valid.min()) if len(valid) else None,
        "end": _iso(valid.max()) if len(valid) else None,
        "missing": int(times.isna().sum()),
        "sorted": bool(valid.is_monotonic_increasing),
        "duplicates": int(valid.duplicated().sum()),
        "median_interval_s": None,
        "irregular_share": None,
        "gaps": [],
        "n_gaps": 0,
    }
    if len(valid) < 3:
        return info
    steps = valid.sort_values().diff().dropna().dt.total_seconds().to_numpy()
    steps = steps[steps > 0]
    if not len(steps):
        return info
    median = float(np.median(steps))
    info["median_interval_s"] = median
    info["irregular_share"] = float(np.mean(np.abs(steps - median) > 0.5 * median))
    ordered = valid.sort_values().reset_index(drop=True)
    gap_mask = ordered.diff().dt.total_seconds().to_numpy() > GAP_FACTOR * median
    gap_idx = np.flatnonzero(gap_mask)
    info["n_gaps"] = int(len(gap_idx))
    largest = sorted(gap_idx, key=lambda i: ordered[i] - ordered[i - 1], reverse=True)[:5]
    info["gaps"] = [
        {
            "start": _iso(ordered[i - 1]),
            "end": _iso(ordered[i]),
            "duration_s": float((ordered[i] - ordered[i - 1]).total_seconds()),
        }
        for i in sorted(largest)
    ]
    return info


def _flat_share(values: np.ndarray) -> tuple[float, int]:
    """(share of readings inside runs of >= STUCK_RUN identical values, longest run)."""
    finite = values[~np.isnan(values)]
    if len(finite) < STUCK_RUN:
        return 0.0, int(len(finite) > 0)
    change = np.empty(len(finite), dtype=bool)
    change[0] = True
    change[1:] = finite[1:] != finite[:-1]
    run_ids = np.cumsum(change)
    lengths = np.bincount(run_ids)[1:]
    longest = int(lengths.max())
    in_long = int(lengths[lengths >= STUCK_RUN].sum())
    return in_long / len(finite), longest


def _outliers(values: np.ndarray) -> int:
    finite = values[~np.isnan(values)]
    if len(finite) < 10:
        return 0
    median = np.median(finite)
    mad = np.median(np.abs(finite - median)) * 1.4826
    if mad == 0:
        return 0
    return int(np.sum(np.abs(finite - median) / mad > OUTLIER_Z))


def _column_profile(name: str, series: pd.Series, *, is_target: bool, is_time: bool, n: int) -> dict[str, Any]:
    missing = int(series.isna().sum())
    share = missing / n if n else 0.0
    issues: list[str] = []
    status = "ok"

    def flag(level: str, text: str) -> None:
        nonlocal status
        issues.append(text)
        if level == "problem" or status == "ok":
            status = level

    numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
    kind = "time" if is_time else "numeric" if numeric else "boolean" if pd.api.types.is_bool_dtype(series) else "text"
    unique = int(series.nunique(dropna=True))
    col: dict[str, Any] = {
        "name": name,
        "dtype": str(series.dtype),
        "kind": kind,
        "missing": missing,
        "missing_share": share,
        "unique": unique,
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
        "flat_share": None,
        "longest_run": None,
        "outliers": None,
        "role": "target" if is_target else "time" if is_time else None,
    }

    if not is_target:
        if share >= MISSING_PROBLEM:
            flag("problem", f"{share:.0%} missing")
        elif share >= MISSING_WARNING:
            flag("warning", f"{share:.0%} missing")
    if n - missing > 0 and unique == 1 and not is_time:
        flag("problem", "constant: every reading is the same")

    if numeric:
        values = series.to_numpy(dtype="float64", na_value=np.nan)
        finite = values[np.isfinite(values)]
        if len(finite):
            col.update(min=_num(finite.min()), max=_num(finite.max()), mean=_num(finite.mean()), std=_num(finite.std()))
        if np.isinf(values).any():
            flag("warning", f"{int(np.isinf(values).sum())} infinite values")
        if not is_target and unique > 10:
            flat, longest = _flat_share(values)
            col.update(flat_share=flat, longest_run=longest)
            if flat >= STUCK_SHARE_WARNING and unique > 1:
                flag("warning", f"possibly stuck: {flat:.0%} of readings in flat runs (longest {longest})")
        n_out = _outliers(values)
        col["outliers"] = n_out
        if n_out:
            flag("warning", f"{n_out} outlier{'s' if n_out != 1 else ''} (beyond {OUTLIER_Z:g} robust SD)")
    elif kind == "text" and not is_time:
        non_null = series.dropna()
        if len(non_null):
            as_num = pd.to_numeric(non_null.astype(str).str.strip(), errors="coerce")
            if as_num.notna().mean() >= 0.9:
                flag("warning", "numbers stored as text: convert the type")
    return {**col, "status": status, "issues": issues}


def _target_profile(df: pd.DataFrame, target: str, time_col: Optional[str], times: Optional[pd.Series]) -> dict[str, Any]:
    present = df[target].notna()
    samples = int(present.sum())
    info: dict[str, Any] = {
        "column": target,
        "samples": samples,
        "share": samples / len(df) if len(df) else 0.0,
        "median_interval_s": None,
        "first": None,
        "last": None,
    }
    if times is not None and samples:
        sample_times = times[present].dropna().sort_values()
        info["first"] = _iso(sample_times.min()) if len(sample_times) else None
        info["last"] = _iso(sample_times.max()) if len(sample_times) else None
        steps = sample_times.diff().dropna().dt.total_seconds()
        steps = steps[steps > 0]
        if len(steps):
            info["median_interval_s"] = float(steps.median())
    return info


def profile_dataframe(df: pd.DataFrame, *, target: Optional[str] = None, time_column: Optional[str] = None) -> dict[str, Any]:
    n = int(len(df))
    target = target if target in df.columns else None
    time_col = detect_time_column(df, time_column)
    times = _as_datetime(df[time_col]) if time_col else None
    columns = [
        _column_profile(str(c), df[c], is_target=c == target, is_time=c == time_col, n=n)
        for c in df.columns
    ]
    counts = {level: sum(1 for c in columns if c["status"] == level) for level in ("ok", "warning", "problem")}
    return {
        "row_count": n,
        "column_count": int(df.shape[1]),
        "memory_bytes": int(df.memory_usage(deep=True).sum()),
        "complete_rows": int(df.notna().all(axis=1).sum()),
        "time_column": time_col,
        "time": _time_profile(times) if times is not None else None,
        "target": _target_profile(df, target, time_col, times) if target else None,
        "columns": columns,
        "summary": counts,
    }
