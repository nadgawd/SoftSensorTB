"""Turning live rows into checked predictions with a compiled deployment bundle."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

from backend.deployment import monitor
from backend.deployment.replay import ROW_ID, Compiled, Op, StepError, coerce_inputs, run_ops

OK = "ok"
WARNING = "warning"
SUPPRESSED = "suppressed"
IGNORED = "ignored"


@dataclass
class Bundle:
    """Everything a deployment needs to predict, saved once when it is created."""

    model: Any
    model_id: str
    algorithm: str
    features: list[str]
    target: str
    ops: list[tuple[int, Op]]
    labels: dict[int, str]
    inputs: list[str]
    input_dtypes: dict[str, str]
    time_column: Optional[str]
    buffer_rows: int
    stats: dict[str, monitor.InputStats]
    region: Optional[monitor.RegionModel]
    test_rmse: Optional[float]

    @property
    def history_rows(self) -> int:
        """Raw rows kept between requests: the steps' look-back, and enough to see a stuck sensor."""
        return max(self.buffer_rows, monitor.STUCK_RUN)

    @property
    def value_inputs(self) -> list[str]:
        return [c for c in self.inputs if c != self.time_column]


def _filtered_rows(compiled: Compiled, cut_data: pd.DataFrame) -> pd.DataFrame:
    """The raw rows that survive the row filters, so a training range leaves out e.g. shutdown periods."""
    frame = coerce_inputs(cut_data[compiled.inputs].copy(), compiled.input_dtypes)
    frame[ROW_ID] = np.arange(len(frame))
    try:
        kept = np.unique(run_ops(compiled.ops, frame, live=True)[ROW_ID].to_numpy()).astype(int)
    except StepError:
        return cut_data
    return cut_data.iloc[kept] if len(kept) else cut_data


def build_bundle(
    compiled: Compiled,
    artifact: dict[str, Any],
    training: pd.DataFrame,
    cut_data: pd.DataFrame,
) -> Bundle:
    features = list(artifact["feature_columns"])
    train_index = [i for i in (artifact.get("train_index") or []) if i in training.index]
    reference = training.loc[train_index, features] if train_index else training[features]
    metrics = artifact.get("metrics") or {}
    return Bundle(
        model=artifact["model"],
        model_id=str(artifact["model_id"]),
        algorithm=str(artifact.get("algorithm") or ""),
        features=features,
        target=str(artifact["target_column"]),
        ops=compiled.ops,
        labels=compiled.labels,
        inputs=list(compiled.inputs),
        input_dtypes=dict(compiled.input_dtypes),
        time_column=compiled.time_column,
        buffer_rows=compiled.buffer_rows,
        stats=monitor.input_stats(_filtered_rows(compiled, cut_data), compiled.inputs, compiled.time_column),
        region=monitor.fit_region(reference),
        test_rmse=_finite(metrics.get("rmse")),
    )


def new_state(gain: float) -> dict[str, Any]:
    return {
        "buffer": [], "last_time": None, "bias": 0.0, "gain": gain,
        "counts": {OK: 0, WARNING: 0, SUPPRESSED: 0, IGNORED: 0},
        "last_received": None,
    }


def _finite(value: Any) -> Optional[float]:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return _finite(value)
    return str(value)


def parse_time(value: Any) -> Optional[pd.Timestamp]:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(stamp):
        return None
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def iso(stamp: pd.Timestamp) -> str:
    return stamp.isoformat()


def predict(
    bundle: Bundle,
    state: dict[str, Any],
    rows: list[dict[str, Any]],
    received_at: Optional[datetime] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Predict for ``rows`` (raw readings, oldest first or with a time); returns records and the new state."""
    received_at = received_at or datetime.now(timezone.utc)
    received = pd.Timestamp(received_at).tz_convert("UTC") if pd.Timestamp(received_at).tzinfo else pd.Timestamp(received_at).tz_localize("UTC")
    state = {**state, "counts": dict(state.get("counts") or {}), "buffer": list(state.get("buffer") or [])}
    last_time = parse_time(state.get("last_time"))

    incoming: list[tuple[pd.Timestamp, int, dict[str, Any]]] = []
    for order, row in enumerate(rows):
        raw = {c: _json_value(row.get(c)) for c in bundle.inputs if c in row}
        stamp = parse_time(row.get(bundle.time_column)) if bundle.time_column else None
        if stamp is None:
            stamp = received + pd.Timedelta(microseconds=order)
            if bundle.time_column:
                raw.setdefault(bundle.time_column, None)
        incoming.append((stamp, order, raw))
    incoming.sort(key=lambda item: (item[0], item[1]))

    records: list[dict[str, Any]] = []
    accepted: list[tuple[pd.Timestamp, dict[str, Any]]] = []
    for stamp, _, raw in incoming:
        if last_time is not None and stamp <= last_time:
            records.append(_record(stamp, received, raw, IGNORED, [
                f"Not newer than the last row received ({iso(last_time)}); a repeated or late row is not predicted again."
            ], state))
            continue
        accepted.append((stamp, raw))
        last_time = stamp

    if accepted:
        records.extend(_predict_accepted(bundle, state, accepted, received))
        state["buffer"] = (state["buffer"] + [raw for _, raw in accepted])[-bundle.history_rows:]
        state["last_time"] = iso(last_time)
    state["last_received"] = iso(received)
    for rec in records:
        state["counts"][rec["status"]] = int(state["counts"].get(rec["status"], 0)) + 1
    records.sort(key=lambda r: r["time"])
    return records, state


def _record(stamp, received, raw, status, reasons, state, prediction=None, t2=None, spe=None) -> dict[str, Any]:
    bias = float(state.get("bias") or 0.0)
    return {
        "time": iso(stamp),
        "received_at": iso(received),
        "prediction": prediction,
        "corrected": None if prediction is None else prediction + bias,
        "bias": bias,
        "status": status,
        "reasons": reasons,
        "t2": t2,
        "spe": spe,
        "inputs": raw,
    }


def _predict_accepted(bundle, state, accepted, received) -> list[dict[str, Any]]:
    history = state["buffer"]
    raws = history + [raw for _, raw in accepted]
    frame = coerce_inputs(pd.DataFrame(raws, columns=bundle.inputs), bundle.input_dtypes)
    first_new = len(history)
    frame[ROW_ID] = np.arange(len(frame))
    new_mask = np.arange(len(frame)) >= first_new
    missing, warnings = monitor.input_findings(frame, bundle.stats, new_mask)

    dropped: dict[int, int] = {}
    try:
        out = run_ops(bundle.ops, frame, live=True, dropped=dropped)
    except StepError as exc:
        reason = f"{bundle.labels.get(exc.index, f'step {exc.index}')} failed on these rows: {exc.error}"
        return [_record(stamp, received, raw, SUPPRESSED, [reason], state) for stamp, raw in accepted]

    out = out.set_index(ROW_ID, drop=False)
    features = out.reindex(columns=bundle.features).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    short_history = len(history) < bundle.buffer_rows

    records: list[Optional[dict[str, Any]]] = [None] * len(accepted)
    valid_rows: list[int] = []
    for i, (stamp, raw) in enumerate(accepted):
        row_id = first_new + i
        reasons = list(warnings[i])
        if row_id in dropped:
            label = bundle.labels.get(dropped[row_id], f"step {dropped[row_id]}")
            records[i] = _record(stamp, received, raw, SUPPRESSED, [
                f"Removed by {label}, as the same rows were in training (for example a shutdown or bad-data filter)."
            ] + reasons, state)
            continue
        if row_id not in features.index:
            records[i] = _record(stamp, received, raw, SUPPRESSED, ["Removed by the preprocessing steps."] + reasons, state)
            continue
        absent = [f for f in bundle.features if pd.isna(features.at[row_id, f])]
        if absent:
            if missing[i]:
                why = f"Missing input: {', '.join(missing[i])}."
            elif short_history:
                why = f"Not enough history yet for {', '.join(absent[:5])} (needs {bundle.buffer_rows} earlier rows)."
            else:
                why = f"Could not compute {', '.join(absent[:5])} from these readings."
            records[i] = _record(stamp, received, raw, SUPPRESSED, [why] + reasons, state)
            continue
        if missing[i]:
            reasons.insert(0, f"Missing input {', '.join(missing[i])} was filled by the preprocessing steps.")
        records[i] = {"_pending": True, "reasons": reasons}
        valid_rows.append(i)

    if valid_rows:
        ids = [first_new + i for i in valid_rows]
        X = features.loc[ids, bundle.features].astype(float)
        try:
            y_hat = np.asarray(bundle.model.predict(X), dtype=float).reshape(-1)
        except Exception as exc:  # noqa: BLE001 - reported on every row
            y_hat = np.full(len(ids), np.nan)
            failure = f"The model could not predict: {exc}"
        else:
            failure = "The model returned no value."
        t2, spe, region_notes = monitor.region_findings(bundle.region, X.to_numpy())
        for j, i in enumerate(valid_rows):
            stamp, raw = accepted[i]
            reasons = records[i]["reasons"] + region_notes[j]
            value = float(y_hat[j])
            if not math.isfinite(value):
                records[i] = _record(stamp, received, raw, SUPPRESSED, [failure] + reasons, state)
                continue
            status = WARNING if reasons else OK
            records[i] = _record(stamp, received, raw, status, reasons, state, prediction=value, t2=t2[j], spe=spe[j])
    return [r for r in records if r is not None]
