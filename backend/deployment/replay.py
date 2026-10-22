"""Replaying a model's preprocessing history on live rows.

A model is trained on a processed dataset version: lag columns, scaled tags,
encoded categories, rows dropped during shutdowns. The plant sends raw tags, so
every step between the upload and the training version has to run again on
each live row, exactly as it ran on the training data.

Steps are rebuilt from the lineage. A step that learnt something from the data
(a scaler, a fill value, outlier fences) is fitted again on its stored parent
version, which reproduces what the tool learnt because versions never change.
Each step is then checked against the stored version it produced, and the whole
chain is checked end to end and in streaming mode before a model may go live.
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from backend.mcp_server import eda_tools as eda
from backend.mcp_server import manipulation_tools as mt
from backend.mcp_server.safe_expr import assert_safe_expression
from backend.services.profile import detect_time_column

# Carried through every step so live rows can be matched to their outputs.
ROW_ID = "__softsensor_row__"
# Any well-formed id: the tools' argument models validate it but the replay never uses it.
_DUMMY_VERSION_ID = "00000000-0000-4000-8000-000000000000"

# How a step behaves on live data.
EXACT = "exact"                # same arithmetic on each row as in training
FITTED = "fitted"              # uses statistics learnt from the training data
WINDOW = "window"              # needs the recent rows (lags, rolling windows)
APPROX = "approximate"         # used later rows in training; live rows only have the past
TRAINING_ONLY = "training_only"  # only shaped the training rows (duplicates, balancing, order)
UPSTREAM = "upstream"          # the live feed must already be in this step's output format
NOT_NEEDED = "not_needed"      # nothing the model uses depends on it
UNSUPPORTED = "unsupported"
NO_CHANGE = "no_change"

_MIN_BUFFER_ROWS = 5
_MAX_BUFFER_ROWS = 5000
_STREAM_CHECK_ROWS = 120
_STREAM_CHUNK = 4


class Unsupported(Exception):
    """This step cannot run on live rows."""


class StepError(Exception):
    def __init__(self, index: int, action: str, error: Exception) -> None:
        super().__init__(f"Step {index} ({action}) failed: {error}")
        self.index = index
        self.action = action


@dataclass
class Context:
    target: str
    time_column: Optional[str]


def _cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != ROW_ID]


def _present(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [c for c in columns if c in df.columns]


def referenced_columns(expression: str, columns: list[str]) -> set[str]:
    """Columns an expression mentions, plain or in backticks."""
    found = set()
    for col in columns:
        name = str(col)
        if f"`{name}`" in expression or re.search(rf"(?<![\w`]){re.escape(name)}(?![\w`])", expression):
            found.add(col)
    return found


def _as_text(series: pd.Series, train_dtype: Any) -> pd.Series:
    """Values as the tools' ``astype("string")`` saw them in training.

    Live numbers arrive as floats; a category code that was the integer 2 in
    training must still read "2", not "2.0".
    """
    if pd.api.types.is_integer_dtype(train_dtype) and pd.api.types.is_float_dtype(series):
        whole = series.notna() & (series % 1 == 0)
        out = series.astype("string")
        out[whole] = series[whole].astype("int64").astype("string")
        return out
    return series.astype("string")


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


class Op:
    """One lineage step, ready to run on a frame of rows."""

    mode = EXACT
    is_filter = False
    lookback = 0
    note = ""

    def __init__(self, params: dict[str, Any], parent: pd.DataFrame, child: pd.DataFrame, ctx: Context) -> None:
        self.params = params
        self.ctx = ctx

    def requires(self, needed: set[str]) -> set[str]:
        """Columns this step reads, given the columns needed after it (in-place columns stay needed anyway)."""
        return set()

    def writes(self) -> set[str]:
        """Existing columns this step changes in place, even if it found nothing to change in training."""
        return set()

    def apply(self, df: pd.DataFrame, live: bool) -> pd.DataFrame:
        raise NotImplementedError


class RenameOp(Op):
    def __init__(self, params, parent, child, ctx):
        super().__init__(params, parent, child, ctx)
        self.mapping = {str(k): str(v) for k, v in (params.get("column_mapping") or {}).items()}

    def requires(self, needed):
        return {old for old, new in self.mapping.items() if new in needed}

    def apply(self, df, live):
        return df.rename(columns={o: n for o, n in self.mapping.items() if o in df.columns})


class FormulaOp(Op):
    def __init__(self, params, parent, child, ctx):
        super().__init__(params, parent, child, ctx)
        self.name = params["new_column_name"]
        self.formula = assert_safe_expression(params["formula"], kind="formula")
        self.inputs = referenced_columns(self.formula, list(parent.columns))

    def requires(self, needed):
        return set(self.inputs) if self.name in needed else set()

    def writes(self):
        return {self.name}

    def apply(self, df, live):
        out = df.copy()
        out[self.name] = out.eval(self.formula)
        return out


class ConvertOp(Op):
    def __init__(self, params, parent, child, ctx):
        super().__init__(params, parent, child, ctx)
        self.columns = list(params["columns"])
        self.args = eda.ConvertTypesArgs(
            dataset_version_id=_DUMMY_VERSION_ID,
            columns=self.columns,
            target_type=params["target_type"],
            errors=params.get("errors", "coerce"),
            datetime_format=params.get("datetime_format"),
        )

    def writes(self):
        return set(self.columns)

    def apply(self, df, live):
        present = _present(df, self.columns)
        if not present:
            return df
        clean = eda._coerce_fake_nulls(df[present])
        out = df.copy()
        for col in present:
            out[col] = eda._convert_column(clean[col], self.args)
        return out


def _condition_args(column: str, operator: str, value: Any) -> eda.DropRowsArgs:
    return eda.DropRowsArgs(
        dataset_version_id=_DUMMY_VERSION_ID, mode="condition", column=column, operator=operator, value=value,
    )


class MaskOp(Op):
    def __init__(self, params, parent, child, ctx):
        super().__init__(params, parent, child, ctx)
        self.columns = list(params["columns"])
        self.action = params.get("action", "mask")
        self.condition_column = params.get("condition_column")
        self.operator = params.get("operator", "greater_than")
        self.value = params.get("value")
        self.replacement = params.get("replacement")

    def requires(self, needed):
        if self.condition_column and any(c in needed for c in self.columns):
            return {self.condition_column} - {self.ctx.target}
        return set()

    def writes(self):
        return set(self.columns)

    def apply(self, df, live):
        out = df.copy()
        for col in _present(df, self.columns):
            tested = self.condition_column or col
            if tested not in df.columns:
                continue
            condition = eda._condition_mask(df, _condition_args(tested, self.operator, self.value))
            replace = condition if self.action == "mask" else ~condition
            original = df[col]
            new_value = mt._typed_replacement(original, self.replacement)
            if pd.api.types.is_integer_dtype(original) and isinstance(new_value, float) and new_value.is_integer():
                new_value = int(new_value)
            if isinstance(original.dtype, pd.CategoricalDtype):
                updated = original.astype(object).mask(replace, new_value).astype("category")
            else:
                updated = original.mask(replace, new_value)
            out[col] = updated
        return out


class MapOp(Op):
    def __init__(self, params, parent, child, ctx):
        super().__init__(params, parent, child, ctx)
        self.columns = list(params["columns"])
        self.keep_unmapped = params.get("unmapped", "keep") == "keep"
        self.lookup = {mt._canonical(k): mt._mapped_target(v) for k, v in (params.get("value_mapping") or {}).items()}

    def writes(self):
        return set(self.columns)

    def apply(self, df, live):
        out = df.copy()
        for col in _present(df, self.columns):
            original = df[col]
            keys = original.astype(object).map(mt._canonical)
            hit = keys.map(lambda k: k in self.lookup).astype(bool)
            mapped = keys.map(lambda k: self.lookup.get(k)).astype(object)
            fallback = original.astype(object) if self.keep_unmapped else None
            chosen = mapped.where(hit, fallback)
            out[col] = pd.Series([None if pd.isna(v) else v for v in chosen], index=original.index)
        return out


_COLUMN_STATS = {"zscore", "minmax", "demean"}


class ApplyOp(Op):
    def __init__(self, params, parent, child, ctx):
        super().__init__(params, parent, child, ctx)
        self.fn = mt.ApplyFunction(params["function"])
        self.columns = list(params["columns"])
        self.group_by = list(params.get("group_by") or [])
        self.decimals = int(params.get("decimals", 2))
        self.in_place = bool(params.get("in_place", False))
        self.stats: dict[str, Any] = {}
        if self.fn in mt._ELEMENTWISE:
            return
        name = self.fn.value
        if name in _COLUMN_STATS and not self.group_by:
            self.mode = FITTED
            for col in self.columns:
                x = parent[col].astype(float)
                self.stats[col] = {"mean": x.mean(), "std": x.std(), "min": x.min(), "max": x.max()}
        elif name.startswith("group_") and self.group_by:
            self.mode = FITTED
            stat = name.removeprefix("group_")
            for col in self.columns:
                frame = parent[self.group_by].copy()
                frame["__value"] = parent[col].astype(float)
                grouped = frame.groupby(self.group_by, dropna=False, observed=True)["__value"]
                self.stats[col] = getattr(grouped, stat)().rename("__stat").reset_index()
        else:
            raise Unsupported(f"{name} depends on every earlier row of the column, which a live feed does not have")

    def _name(self, col: str) -> str:
        if self.in_place:
            return col
        return f"{col}_{self.fn.value}" + (f"_by_{'_'.join(self.group_by)}" if self.group_by else "")

    def requires(self, needed):
        out: set[str] = set()
        for col in self.columns:
            if self._name(col) in needed:
                out |= {col, *self.group_by}
        return out

    def writes(self):
        return {self._name(col) for col in self.columns}

    def apply(self, df, live):
        out = df.copy()
        for col in _present(df, self.columns):
            x = df[col].astype(float)
            name = self.fn.value
            if self.fn in mt._ELEMENTWISE:
                values = mt._elementwise(x, self.fn, self.decimals)
            elif name in _COLUMN_STATS:
                s = self.stats[col]
                if name == "zscore":
                    std = s["std"]
                    values = (x - s["mean"]) / std if std and np.isfinite(std) else x * np.nan
                elif name == "minmax":
                    span = s["max"] - s["min"]
                    values = (x - s["min"]) / span if span and np.isfinite(span) else x * np.nan
                else:
                    values = x - s["mean"]
            else:
                if not all(g in df.columns for g in self.group_by):
                    continue
                keys = df[self.group_by].reset_index(drop=True)
                values = keys.merge(self.stats[col], how="left", on=self.group_by)["__stat"]
            out[self._name(col)] = np.asarray(values, dtype=float)
        return out


class MissingOp(Op):
    def __init__(self, params, parent, child, ctx):
        super().__init__(params, parent, child, ctx)
        self.strategy = params["strategy"]
        self.columns = params.get("columns")
        clean = eda._coerce_fake_nulls(parent)
        target_cols = list(self.columns or _cols(clean))
        self.targets = target_cols
        self.flagged = [c for c in target_cols if clean[c].isna().any()] if params.get("flag_missing") else []
        numeric = [c for c in target_cols if pd.api.types.is_numeric_dtype(clean[c])]
        self.fill: dict[str, Any] = {}
        self.imputer = None
        self.numeric_cols = numeric
        self.args = None
        s = self.strategy
        if s == "drop_rows":
            self.is_filter = True
        elif s in ("fill_mean", "fill_median"):
            self.mode = FITTED
            stats = clean[numeric].mean() if s == "fill_mean" else clean[numeric].median()
            self.fill = stats.to_dict()
        elif s == "fill_mode":
            self.mode = FITTED
            for col in target_cols:
                modes = clean[col].mode()
                if not modes.empty:
                    self.fill[col] = modes.iloc[0]
        elif s == "forward_fill":
            self.mode = WINDOW
            self.note = "Fills a gap with the last reading in the recent history."
        elif s == "backward_fill":
            self.mode = APPROX
            self.note = "Training filled gaps from the next reading; a live gap stays empty until a later row arrives."
        elif s in ("knn", "iterative"):
            self.mode = FITTED
            from sklearn.experimental import enable_iterative_imputer  # noqa: F401
            from sklearn.impute import IterativeImputer, KNNImputer

            if numeric:
                if s == "knn":
                    self.imputer = KNNImputer(n_neighbors=int(params.get("n_neighbors", 5)))
                else:
                    self.imputer = IterativeImputer(
                        max_iter=int(params.get("max_iter", 10)), random_state=int(params.get("random_state", 42))
                    )
                self.imputer.fit(clean[numeric])
        elif s == "interpolate":
            self.mode = APPROX
            self.note = "Training interpolated between the readings around a gap; live, the newest gap is filled from earlier readings only."
            self.args = eda.RemoveMissingDataArgs(
                dataset_version_id=_DUMMY_VERSION_ID,
                strategy="interpolate",
                columns=self.columns,
                interpolation_method=params.get("interpolation_method", "linear"),
                interpolation_order=int(params.get("interpolation_order", 3)),
                limit_direction=params.get("limit_direction", "both"),
                time_column=params.get("time_column"),
            )
        else:
            raise Unsupported(f"unknown missing-data strategy {s!r}")

    def requires(self, needed):
        req = {c for c in self.flagged if f"{c}_is_missing" in needed}
        if self.strategy == "drop_rows" and self.columns:
            req |= set(self.columns)
        if self.imputer is not None and any(c in needed for c in self.numeric_cols):
            req |= set(self.numeric_cols)
        if self.args is not None and self.args.interpolation_method.value == "time" and self.args.time_column:
            req.add(self.args.time_column)
        return req - {self.ctx.target}

    def writes(self):
        return set() if self.strategy == "drop_rows" else set(self.targets)

    def apply(self, df, live):
        out = eda._coerce_fake_nulls(df)
        target_cols = _present(out, list(self.columns or _cols(out)))
        for col in self.flagged:
            if col in out.columns:
                out[f"{col}_is_missing"] = out[col].isna().astype(int)
        numeric = [c for c in target_cols if pd.api.types.is_numeric_dtype(out[c])]
        s = self.strategy
        if s == "drop_rows":
            return out.dropna(subset=target_cols).reset_index(drop=True)
        if s in ("fill_mean", "fill_median"):
            cols = [c for c in numeric if c in self.fill]
            if cols:
                out[cols] = out[cols].fillna(pd.Series({c: self.fill[c] for c in cols}))
        elif s == "fill_mode":
            for col in target_cols:
                if col in self.fill:
                    out[col] = out[col].fillna(self.fill[col])
        elif s == "forward_fill":
            out[target_cols] = out[target_cols].ffill()
        elif s == "backward_fill":
            out[target_cols] = out[target_cols].bfill()
        elif s in ("knn", "iterative") and self.imputer is not None:
            frame = out.reindex(columns=self.numeric_cols).astype(float)
            filled = self.imputer.transform(frame)
            for i, col in enumerate(self.numeric_cols):
                if col in out.columns:
                    out[col] = filled[:, i]
        elif s == "interpolate" and numeric:
            try:
                out = eda._interpolate(out, numeric, self.args)
            except ValueError:
                if not live:
                    raise
        return out


class OutlierOp(Op):
    def __init__(self, params, parent, child, ctx):
        super().__init__(params, parent, child, ctx)
        self.method = params["method"]
        self.columns = list(params["columns"])
        self.bounds: dict[str, tuple[Optional[float], Optional[float]]] = {}
        if self.method == "clip":
            for col in self.columns:
                self.bounds[col] = (params.get("lower_bound"), params.get("upper_bound"))
            return
        self.mode = FITTED
        threshold = float(params["threshold"])
        if self.method == "winsorize":
            from scipy.stats import mstats

            for col in self.columns:
                capped = np.asarray(mstats.winsorize(parent[col], limits=[threshold, threshold]), dtype=float)
                self.bounds[col] = (float(np.nanmin(capped)), float(np.nanmax(capped)))
            return
        for col in self.columns:
            series = parent[col]
            if self.method == "z_score":
                std = series.std(ddof=0)
                if std == 0 or pd.isna(std):
                    continue
                mean = series.mean()
                self.bounds[col] = (mean - threshold * std, mean + threshold * std)
            else:
                q1, q3 = series.quantile(0.25), series.quantile(0.75)
                iqr = q3 - q1
                self.bounds[col] = (q1 - threshold * iqr, q3 + threshold * iqr)
        self.is_filter = params.get("action", "drop") == "drop"

    def requires(self, needed):
        return set(self.bounds) - {self.ctx.target} if self.is_filter else set()

    def writes(self):
        return set() if self.is_filter else set(self.bounds)

    def apply(self, df, live):
        out = df.copy()
        if self.is_filter:
            keep = pd.Series(True, index=out.index)
            for col, (lower, upper) in self.bounds.items():
                if col in out.columns:
                    keep &= out[col].between(lower, upper)
            return out.loc[keep].reset_index(drop=True)
        for col, (lower, upper) in self.bounds.items():
            if col in out.columns:
                out[col] = out[col].clip(lower=lower, upper=upper)
        return out


_SCALERS = {
    "standard": lambda p: __import__("sklearn.preprocessing", fromlist=["StandardScaler"]).StandardScaler(),
    "min_max": lambda p: __import__("sklearn.preprocessing", fromlist=["MinMaxScaler"]).MinMaxScaler(
        feature_range=tuple(p.get("feature_range") or (0.0, 1.0))
    ),
    "robust": lambda p: __import__("sklearn.preprocessing", fromlist=["RobustScaler"]).RobustScaler(
        quantile_range=tuple(p.get("quantile_range") or (25.0, 75.0))
    ),
    "maxabs": lambda p: __import__("sklearn.preprocessing", fromlist=["MaxAbsScaler"]).MaxAbsScaler(),
}


class NormalizeOp(Op):
    mode = FITTED

    def __init__(self, params, parent, child, ctx):
        super().__init__(params, parent, child, ctx)
        make = _SCALERS.get(params.get("strategy", "standard"))
        if make is None:
            raise Unsupported(f"unknown scaling strategy {params.get('strategy')!r}")
        # Every one of these scalers treats each column on its own, so one per column
        # gives the same numbers and lets a live feed send only the columns in use.
        self.scalers = {col: make(params).fit(parent[[col]]) for col in params["columns"]}

    def writes(self):
        return set(self.scalers)

    def apply(self, df, live):
        out = df.copy()
        for col, scaler in self.scalers.items():
            if col in out.columns:
                out[col] = scaler.transform(out[[col]]).ravel()
        return out


_TRANSFORM_SUFFIX = {
    "log": "log", "sqrt": "sqrt", "reciprocal": "reciprocal", "square": "square",
    "box_cox": "boxcox", "yeo_johnson": "yeo", "binarize": "bin",
    "bin_equal_width": "binned", "bin_equal_freq": "binned",
}


class TransformOp(Op):
    def __init__(self, params, parent, child, ctx):
        super().__init__(params, parent, child, ctx)
        self.strategy = params["strategy"]
        self.columns = list(params["columns"])
        if self.strategy not in _TRANSFORM_SUFFIX:
            raise Unsupported(f"unknown transform {self.strategy!r}")
        self.suffix = _TRANSFORM_SUFFIX[self.strategy]
        self.fitted: dict[str, Any] = {}
        if self.strategy == "box_cox":
            from scipy.stats import boxcox

            self.mode = FITTED
            for col in self.columns:
                val = parent[col]
                shift = float(1 - val.min()) if (val <= 0).any() else 0.0
                _, lmbda = boxcox(val + shift if shift else val)
                self.fitted[col] = (shift, float(lmbda))
        elif self.strategy == "yeo_johnson":
            from scipy.stats import yeojohnson

            self.mode = FITTED
            for col in self.columns:
                _, lmbda = yeojohnson(parent[col])
                self.fitted[col] = float(lmbda)
        elif self.strategy in ("bin_equal_width", "bin_equal_freq"):
            from sklearn.preprocessing import KBinsDiscretizer

            self.mode = FITTED
            kind = "uniform" if self.strategy == "bin_equal_width" else "quantile"
            for col in self.columns:
                self.fitted[col] = KBinsDiscretizer(
                    n_bins=int(params.get("n_bins", 5)), encode="ordinal", strategy=kind
                ).fit(parent[[col]])
        elif self.strategy == "binarize":
            self.threshold = float(params.get("threshold", 0.0))

    def requires(self, needed):
        return {c for c in self.columns if f"{c}_{self.suffix}" in needed}

    def apply(self, df, live):
        from scipy.special import boxcox as boxcox_with
        from scipy.stats import yeojohnson

        out = df.copy()
        for col in _present(df, self.columns):
            name = f"{col}_{self.suffix}"
            s = self.strategy
            if s == "log":
                val = out[col].astype(float)
                out[name] = np.log1p(val.mask(val <= -1))
            elif s == "sqrt":
                out[name] = np.sqrt(out[col].clip(lower=0))
            elif s == "reciprocal":
                out[name] = 1.0 / out[col].astype(float).replace(0, np.nan)
            elif s == "square":
                out[name] = out[col].astype(float) ** 2
            elif s == "box_cox":
                shift, lmbda = self.fitted[col]
                with np.errstate(all="ignore"):
                    values = boxcox_with(out[col].astype(float) + shift, lmbda)
                out[name] = pd.Series(values, index=out.index).replace([np.inf, -np.inf], np.nan)
            elif s == "yeo_johnson":
                out[name] = yeojohnson(out[col].astype(float).to_numpy(), self.fitted[col])
            elif s == "binarize":
                x = out[col]
                flags = (x > self.threshold).astype(x.dtype if pd.api.types.is_numeric_dtype(x) else float)
                out[name] = flags.where(x.notna())
            else:
                known = out[col].notna()
                values = pd.Series(np.nan, index=out.index)
                if known.any():
                    values[known] = self.fitted[col].transform(out.loc[known, [col]]).ravel()
                out[name] = values
        return out


class EncodeOp(Op):
    def __init__(self, params, parent, child, ctx):
        super().__init__(params, parent, child, ctx)
        self.strategy = params["strategy"]
        self.columns = list(params["columns"])
        self.drop_first = bool(params.get("drop_first", True))
        self.dtypes = {c: parent[c].dtype for c in self.columns}
        self.maps: dict[str, Any] = {}
        self.names: dict[str, list[str]] = {}
        s = self.strategy
        if s == "one_hot":
            self.mode = FITTED
            for col in self.columns:
                cats = list(pd.Categorical(parent[col]).categories)
                self.maps[col] = cats
                empty = pd.DataFrame({col: pd.Categorical([], categories=cats)})
                self.names[col] = list(pd.get_dummies(empty, columns=[col], drop_first=self.drop_first).columns)
        elif s in ("label", "ordinal"):
            order = params.get("category_order")
            if order:
                self.maps["__order__"] = {category: i for i, category in enumerate(order, start=1)}
            else:
                self.mode = FITTED
                for col in self.columns:
                    _, uniques = pd.factorize(parent[col].astype("string"), sort=True)
                    start = 0 if s == "label" else 1
                    self.maps[col] = {str(u): i for i, u in enumerate(uniques, start=start)}
        elif s == "frequency":
            self.mode = FITTED
            for col in self.columns:
                self.maps[col] = parent[col].value_counts(normalize=True)
        elif s == "binary":
            self.mode = FITTED
            for col in self.columns:
                _, uniques = pd.factorize(parent[col].astype("string"), sort=True)
                codes = {str(u): i for i, u in enumerate(uniques, start=1)}
                n_bits = max(1, len(uniques).bit_length()) if len(uniques) else 1
                self.maps[col] = (codes, n_bits)
                self.names[col] = [f"{col}_{i}" for i in range(n_bits)]
        elif s == "target":
            self.mode = FITTED
            y = parent[params["target_column"]].astype(float)
            smoothing = float(params.get("smoothing", 1.0))
            overall = float(y.mean())
            for col in self.columns:
                stats = y.groupby(parent[col]).agg(["mean", "count"])
                encoded = (stats["count"] * stats["mean"] + smoothing * overall) / (stats["count"] + smoothing)
                self.maps[col] = (encoded, overall)
        elif s == "leave_one_out":
            self.mode = APPROX
            self.note = "Training used the target mean of the other rows in the category; live rows use the category's mean over all training rows."
            self.loo_target = params["target_column"]
            y = parent[self.loo_target].astype(float)
            overall = float(y.mean())
            for col in self.columns:
                self.maps[col] = (y.groupby(parent[col]).mean(), overall)
        elif s == "woe":
            self.mode = FITTED
            event, _ = eda._binary_target(parent[params["target_column"]], params.get("positive_class"))
            for col in self.columns:
                self.maps[col] = self._woe_table(parent[col], event)
        else:
            raise Unsupported(f"unknown encoding {s!r}")

    @staticmethod
    def _woe_table(series: pd.Series, event: pd.Series, regularization: float = 1.0) -> pd.Series:
        # The same arithmetic as eda_tools._woe_encode, kept as a lookup table.
        labelled = event.notna()
        cats, y = series[labelled], event[labelled]
        total_events = float(y.sum())
        total_non = float(len(y) - total_events)
        stats = y.groupby(cats).agg(["sum", "count"])
        share_event = (stats["sum"] + regularization) / (total_events + 2 * regularization)
        share_non = (stats["count"] - stats["sum"] + regularization) / (total_non + 2 * regularization)
        return np.log(share_event / share_non)

    def requires(self, needed):
        if self.strategy in ("one_hot", "binary"):
            return {c for c in self.columns if any(n in needed for n in self.names[c])}
        if self.strategy == "frequency":
            return {c for c in self.columns if f"{c}_freq" in needed}
        return set()

    def apply(self, df, live):
        out = df.copy()
        present = _present(out, self.columns)
        s = self.strategy
        if s == "one_hot":
            for col in present:
                out[col] = pd.Categorical(out[col], categories=self.maps[col])
            return pd.get_dummies(out, columns=present, drop_first=self.drop_first) if present else out
        for col in present:
            if s in ("label", "ordinal"):
                mapping = self.maps.get("__order__") or self.maps[col]
                text = _as_text(out[col], self.dtypes[col])
                out[col] = text.map(lambda v: mapping.get(v, np.nan) if isinstance(v, str) else np.nan).astype(float)
            elif s == "frequency":
                out[f"{col}_freq"] = out[col].map(self.maps[col])
            elif s == "binary":
                codes, n_bits = self.maps[col]
                text = _as_text(out[col], self.dtypes[col])
                ints = text.map(lambda v: codes.get(v, 0) if isinstance(v, str) else 0).astype(int).to_numpy()
                position = out.columns.get_loc(col)
                out = out.drop(columns=[col])
                for offset, i in enumerate(range(n_bits)):
                    out.insert(position + offset, f"{col}_{i}", (ints >> (n_bits - 1 - i)) & 1)
            elif s == "target":
                encoded, overall = self.maps[col]
                out[col] = out[col].map(encoded).astype(float).fillna(overall)
            elif s == "leave_one_out":
                if not live and self.loo_target in out.columns:
                    out[col] = eda._leave_one_out_encode(out[col], out[self.loo_target].astype(float))
                else:
                    means, overall = self.maps[col]
                    out[col] = out[col].map(means).astype(float).fillna(overall)
            elif s == "woe":
                out[col] = out[col].map(self.maps[col]).astype(float).fillna(0.0)
        return out


class ReduceOp(Op):
    mode = FITTED

    def __init__(self, params, parent, child, ctx):
        super().__init__(params, parent, child, ctx)
        self.strategy = params["strategy"]
        self.columns = list(params["columns"])
        self.keep_original = bool(params.get("keep_original", False))
        k = int(params.get("n_components") or 2)
        s = self.strategy
        if s == "tsne":
            raise Unsupported("t-SNE only embeds the rows it was fitted on; it cannot place new rows")
        if s == "pca":
            from sklearn.decomposition import PCA

            self.prefix = "pca"
            X = parent[self.columns].fillna(0)
            self.model = PCA(n_components=k).fit(X)
            names = [f"pca_{i}" for i in range(k)]
            self.linear = None
            if all(n in child.columns for n in names) and len(child) == len(parent):
                # A randomised solver can land on different components when refitted;
                # the stored components then define the projection.
                if not np.allclose(self.model.transform(X), child[names].to_numpy(dtype=float), rtol=1e-6, atol=1e-8):
                    design = np.column_stack([X.to_numpy(dtype=float), np.ones(len(X))])
                    self.linear, *_ = np.linalg.lstsq(design, child[names].to_numpy(dtype=float), rcond=None)
            self.names = names
            return
        if s not in eda._COMPONENT_PREFIX:
            raise Unsupported(f"unknown reduction {s!r}")
        from sklearn.preprocessing import StandardScaler

        self.prefix = eda._COMPONENT_PREFIX[eda.ReduceDimensionsStrategy(s)]
        needed = list(self.columns) + ([params["target_column"]] if s == "lda" else [])
        complete = parent[needed].dropna()
        X = complete[self.columns].astype(float).to_numpy()
        self.scaler = None
        if s == "truncated_svd":
            from sklearn.decomposition import TruncatedSVD

            self.model = TruncatedSVD(n_components=k, random_state=int(params.get("random_state", 42))).fit(X)
        elif s == "lda":
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

            self.scaler = StandardScaler().fit(X)
            labels = complete[params["target_column"]].astype(str).to_numpy()
            self.model = LinearDiscriminantAnalysis(n_components=k).fit(self.scaler.transform(X), labels)
        else:
            from sklearn.exceptions import ConvergenceWarning
            from sklearn.neural_network import MLPRegressor

            self.scaler = StandardScaler().fit(X)
            scaled = self.scaler.transform(X)
            hidden = int(params.get("hidden_units", 16))
            self.model = MLPRegressor(
                hidden_layer_sizes=(hidden, k, hidden), activation="tanh",
                max_iter=int(params.get("max_iter", 500)), random_state=int(params.get("random_state", 42)),
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                self.model.fit(scaled, scaled)
        self.names = [f"{self.prefix}_{i}" for i in range(k)]

    def requires(self, needed):
        return set(self.columns) if any(n in needed for n in self.names) else set()

    def _components(self, X: np.ndarray) -> np.ndarray:
        if self.strategy == "pca":
            if self.linear is not None:
                return np.column_stack([X, np.ones(len(X))]) @ self.linear
            return self.model.transform(X)
        if self.strategy == "truncated_svd":
            return self.model.transform(X)
        scaled = self.scaler.transform(X)
        if self.strategy == "lda":
            return self.model.transform(scaled)
        net = self.model
        hidden = np.tanh(scaled @ net.coefs_[0] + net.intercepts_[0])
        return np.tanh(hidden @ net.coefs_[1] + net.intercepts_[1])

    def apply(self, df, live):
        out = df.copy()
        if not all(c in out.columns for c in self.columns):
            return out
        if self.strategy == "pca":
            frame = out[self.columns].fillna(0)
            comps = self._components(frame if self.linear is None else frame.to_numpy(dtype=float))
            for i, name in enumerate(self.names):
                out[name] = comps[:, i]
        else:
            complete = out[self.columns].notna().all(axis=1)
            target = self.params.get("target_column")
            if self.strategy == "lda" and not live and target in out.columns:
                complete &= out[target].notna()
            for name in self.names:
                out[name] = np.nan
            if complete.any():
                comps = self._components(out.loc[complete, self.columns].astype(float).to_numpy())
                for i, name in enumerate(self.names):
                    out.loc[complete, name] = comps[:, i]
        if not self.keep_original:
            out = out.drop(columns=self.columns)
        return out


class EngineerOp(Op):
    def __init__(self, params, parent, child, ctx):
        super().__init__(params, parent, child, ctx)
        self.strategy = params["strategy"]
        self.columns = list(params["columns"])
        self.created = [c for c in child.columns if c not in parent.columns]
        self.cycles: dict[str, float] = {}
        if self.strategy == "cyclical":
            period = params.get("period")
            for col in self.columns:
                self.cycles[col] = float(period) if period is not None else float(parent[col].max())
            if params.get("period") is None:
                self.mode = FITTED
        elif self.strategy not in ("polynomial", "datetime", "pairwise", "ratio"):
            raise Unsupported(f"unknown feature strategy {self.strategy!r}")

    def requires(self, needed):
        if self.strategy in ("polynomial", "ratio"):
            return set(self.columns) if any(n in needed for n in self.created) else set()
        if self.strategy == "pairwise":
            symbol = eda._PAIR_SYMBOLS[eda.PairOperation(self.params.get("pair_operation", "product"))]
            out: set[str] = set()
            for i, a in enumerate(self.columns):
                for b in self.columns[i + 1:]:
                    if f"{a}_{symbol}_{b}" in needed:
                        out |= {a, b}
            return out
        return {c for c in self.columns if any(n == c or n.startswith(f"{c}_") for n in needed if n in self.created or n == c)}

    def apply(self, df, live):
        out = df.copy()
        if not all(c in out.columns for c in self.columns):
            return out
        s = self.strategy
        if s == "polynomial":
            from sklearn.preprocessing import PolynomialFeatures

            poly = PolynomialFeatures(
                degree=int(self.params.get("degree", 2)),
                interaction_only=bool(self.params.get("interaction_only", False)),
                include_bias=False,
            )
            values = poly.fit_transform(out[self.columns].fillna(0))
            for i, name in enumerate(poly.get_feature_names_out(self.columns)):
                if name not in df.columns:
                    out[name] = values[:, i]
        elif s == "datetime":
            country = self.params.get("holiday_country")
            for col in self.columns:
                stamps = out[col]
                if not pd.api.types.is_datetime64_any_dtype(stamps):
                    stamps = (
                        pd.to_datetime(stamps, errors="coerce")
                        if eda._is_plain_numeric(stamps)
                        else pd.to_datetime(stamps, errors="coerce", format="mixed")
                    )
                out[col] = stamps
                for name, values in eda._datetime_parts(stamps, col, country).items():
                    out[name] = values
        elif s == "pairwise":
            op = eda.PairOperation(self.params.get("pair_operation", "product"))
            for i, a in enumerate(self.columns):
                for b in self.columns[i + 1:]:
                    x, y = out[a].astype(float), out[b].astype(float)
                    values = {
                        eda.PairOperation.product: lambda: x * y,
                        eda.PairOperation.sum: lambda: x + y,
                        eda.PairOperation.difference: lambda: x - y,
                        eda.PairOperation.ratio: lambda: x / y.replace(0, np.nan),
                    }[op]()
                    out[f"{a}_{eda._PAIR_SYMBOLS[op]}_{b}"] = values
        elif s == "cyclical":
            for col in self.columns:
                cycle = self.cycles[col]
                if cycle > 0:
                    out[f"{col}_sin"] = np.sin(2 * np.pi * out[col] / cycle)
                    out[f"{col}_cos"] = np.cos(2 * np.pi * out[col] / cycle)
        else:
            a, b = self.columns[0], self.columns[1]
            out[f"{a}_ratio_{b}"] = out[a] / out[b].replace(0, np.nan)
        return out


class RollingOp(Op):
    def __init__(self, params, parent, child, ctx):
        super().__init__(params, parent, child, ctx)
        self.kind = params.get("window_type", "rolling")
        self.columns = list(params["columns"])
        self.func = params.get("agg_func", "mean")
        self.min_periods = int(params.get("min_periods", 1))
        if self.kind == "expanding":
            raise Unsupported("an expanding window depends on every row since the first one")
        if self.kind == "rolling":
            self.window = int(params["window"])
            self.center = bool(params.get("center", False))
            self.lookback = self.window
            self.mode = APPROX if self.center else WINDOW
            if self.center:
                self.note = "A centred window used later rows in training; live rows only have the earlier ones."
        else:
            self.span = float(params["span"])
            self.lookback = max(int(math.ceil(10 * self.span)), 20)
            self.mode = WINDOW

    def _name(self, col: str) -> str:
        if self.kind == "rolling":
            return f"{col}_rolling{self.window}_{self.func}"
        return f"{col}_ewm{self.span:g}_{self.func}"

    def requires(self, needed):
        return {c for c in self.columns if self._name(c) in needed}

    def apply(self, df, live):
        out = df.copy()
        for col in _present(out, self.columns):
            if self.kind == "rolling":
                roller = out[col].rolling(window=self.window, min_periods=self.min_periods, center=self.center)
            else:
                roller = out[col].ewm(span=self.span, min_periods=self.min_periods)
            out[self._name(col)] = getattr(roller, self.func)()
        return out


class LagOp(Op):
    mode = WINDOW

    def __init__(self, params, parent, child, ctx):
        super().__init__(params, parent, child, ctx)
        self.operation = params.get("operation", "lag")
        self.columns = list(params["columns"])
        self.periods = [int(p) for p in params["lag_periods"]]
        if self.operation == "lead":
            raise Unsupported("a lead column holds later readings, which do not exist yet when a live row arrives")
        self.lookback = max(self.periods)

    def requires(self, needed):
        return {c for c in self.columns if any(f"{c}_{self.operation}{p}" in needed for p in self.periods)}

    def apply(self, df, live):
        out = df.copy()
        for col in _present(out, self.columns):
            for lag in self.periods:
                if self.operation == "lag":
                    values = out[col].shift(lag)
                elif self.operation == "diff":
                    values = out[col].diff(lag)
                else:
                    values = out[col].pct_change(lag, fill_method=None).replace([np.inf, -np.inf], np.nan)
                out[f"{col}_{self.operation}{lag}"] = values
        return out


class DropRowsOp(Op):
    is_filter = True

    def __init__(self, params, parent, child, ctx):
        super().__init__(params, parent, child, ctx)
        self.mode_name = params["mode"]
        self.keep_matching = bool(params.get("keep_matching", False))
        if self.mode_name == "query":
            self.query = assert_safe_expression(params["query"], kind="query")
            self.inputs = referenced_columns(self.query, list(parent.columns))
        else:
            self.args = _condition_args(params["column"], params.get("operator", "equals"), params.get("value"))
            self.inputs = {params["column"]}
        if ctx.target in self.inputs:
            self.note = f"Tests the target {ctx.target!r}, which live rows do not have, so it only shaped the training rows."

    def requires(self, needed):
        return set() if self.ctx.target in self.inputs else set(self.inputs)

    def apply(self, df, live):
        if not all(c in df.columns for c in self.inputs):
            return df
        matching = eda.query_mask(df, self.query) if self.mode_name == "query" else eda._condition_mask(df, self.args)
        drop = ~matching if self.keep_matching else matching
        return df.loc[~drop].reset_index(drop=True)


class AnomalyOp(Op):
    mode = FITTED

    def __init__(self, params, parent, child, ctx):
        super().__init__(params, parent, child, ctx)
        from sklearn.ensemble import IsolationForest

        self.columns = list(params["columns"])
        self.drop = bool(params.get("drop_anomalies", True))
        self.is_filter = self.drop
        self.forest = IsolationForest(
            n_estimators=int(params.get("n_estimators", 100)),
            contamination=params.get("contamination", 0.05),
            random_state=int(params.get("random_state", 42)),
            n_jobs=-1,
        ).fit(parent[self.columns].fillna(0))
        self.uses_target = ctx.target in self.columns
        if self.uses_target:
            if not self.drop:
                raise Unsupported(f"the anomaly model was fitted with the target {ctx.target!r}, which live rows do not have")
            self.note = f"The anomaly model includes the target {ctx.target!r}, so it only shaped the training rows."

    def requires(self, needed):
        if self.uses_target:
            return set()
        if self.drop or "is_anomaly" in needed:
            return set(self.columns)
        return set()

    def apply(self, df, live):
        if not all(c in df.columns for c in self.columns):
            return df
        anomalous = self.forest.predict(df[self.columns].fillna(0)) == -1
        if self.drop:
            return df.loc[~anomalous].reset_index(drop=True)
        out = df.copy()
        out["is_anomaly"] = anomalous.astype(int)
        return out


_OPS: dict[str, type[Op]] = {
    "rename_columns": RenameOp,
    "execute_formula": FormulaOp,
    "convert_types": ConvertOp,
    "mask_values": MaskOp,
    "map_values": MapOp,
    "apply_function": ApplyOp,
    "remove_missing_data": MissingOp,
    "remove_outliers": OutlierOp,
    "normalize_data": NormalizeOp,
    "transform_features": TransformOp,
    "encode_categorical": EncodeOp,
    "reduce_dimensions": ReduceOp,
    "feature_engineer": EngineerOp,
    "rolling_aggregate": RollingOp,
    "create_lag_features": LagOp,
    "drop_rows": DropRowsOp,
    "detect_anomalies": AnomalyOp,
}

_UPSTREAM_ACTIONS = {
    "resample_time_series": "Resampling changes which rows exist; the live feed must send rows at the resampled interval (for example a historian's interpolated or averaged values).",
    "reshape_data": "Reshaping changes the table's layout; the live feed must send rows in the reshaped layout.",
}
_TRAINING_ONLY_NOTES = {
    "drop_duplicates": "Removed duplicate training rows; live rows are scored one by one.",
    "balance_data": "Resampled the training rows to balance classes; it has no live counterpart.",
    "sort_data": "Put the training rows in order; live rows must arrive in the same order (oldest first).",
}

ACTION_TITLES = {
    "initial_upload": "Upload",
    "drop_columns": "Drop columns",
    "rename_columns": "Rename columns",
    "execute_formula": "Formula column",
    "convert_types": "Convert types",
    "mask_values": "Replace values by condition",
    "map_values": "Map values",
    "apply_function": "Apply function",
    "remove_missing_data": "Handle missing values",
    "remove_outliers": "Handle outliers",
    "normalize_data": "Scale columns",
    "transform_features": "Transform columns",
    "encode_categorical": "Encode categories",
    "reduce_dimensions": "Reduce dimensions",
    "feature_engineer": "Engineer features",
    "rolling_aggregate": "Rolling window",
    "create_lag_features": "Lag features",
    "drop_rows": "Drop rows",
    "detect_anomalies": "Detect anomalies",
    "drop_duplicates": "Drop duplicates",
    "balance_data": "Balance classes",
    "sort_data": "Sort rows",
    "resample_time_series": "Resample",
    "reshape_data": "Reshape",
    "combine_datasets": "Combine datasets",
    "select_features_rfe": "Select features (RFE)",
    "select_k_best": "Select features (K best)",
    "select_features_model": "Select features (model)",
    "drop_collinear_features": "Drop collinear features",
    "clean_text": "Clean text",
}


def action_title(action: str) -> str:
    if action.startswith("trained_model_for_"):
        return f"Model predictions for {action.removeprefix('trained_model_for_')}"
    return ACTION_TITLES.get(action, action.replace("_", " ").capitalize())


def _params_summary(action: str, params: dict[str, Any]) -> str:
    keys = ("strategy", "method", "operation", "function", "mode", "window_type", "columns", "lag_periods",
            "window", "span", "formula", "new_column_name", "query", "column", "operator", "value", "frequency", "by")
    parts = []
    for key in keys:
        value = params.get(key)
        if value in (None, "", []):
            continue
        if isinstance(value, list):
            value = ", ".join(map(str, value[:6])) + (" …" if len(value) > 6 else "")
        parts.append(f"{key}: {value}")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Compiling a lineage
# ---------------------------------------------------------------------------


@dataclass
class VersionInfo:
    id: str
    action: str
    params: dict[str, Any]
    load: Callable[[], pd.DataFrame]


@dataclass
class StepReport:
    index: int
    action: str
    title: str
    summary: str
    version_id: str
    mode: str
    note: str = ""
    kept: bool = False
    verified: Optional[bool] = None
    detail: str = ""
    created: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    rows_before: int = 0
    rows_after: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index, "action": self.action, "title": self.title, "summary": self.summary,
            "version_id": self.version_id, "mode": self.mode, "note": self.note, "kept": self.kept,
            "verified": self.verified, "detail": self.detail,
            "created": self.created[:30], "removed": self.removed[:30], "modified": self.modified[:30],
            "rows_before": self.rows_before, "rows_after": self.rows_after,
        }


@dataclass
class Compiled:
    steps: list[StepReport]
    ops: list[tuple[int, Op]]
    inputs: list[str]
    input_dtypes: dict[str, str]
    time_column: Optional[str]
    buffer_rows: int
    cut_index: int
    cut_version_id: str
    blocking: list[str]
    warnings: list[str]
    checks: dict[str, Any]
    approx_features: list[str]
    labels: dict[int, str]


def _diff(parent: pd.DataFrame, child: pd.DataFrame) -> tuple[list[str], list[str], list[str], bool]:
    created = [c for c in child.columns if c not in parent.columns]
    removed = [c for c in parent.columns if c not in child.columns]
    rows_changed = len(parent) != len(child)
    modified: list[str] = []
    if not rows_changed:
        for col in child.columns:
            if col not in parent.columns:
                continue
            if str(parent[col].dtype) != str(child[col].dtype) or not same_values(parent[col], child[col])[0]:
                modified.append(col)
    return created, removed, modified, rows_changed


def same_values(a: pd.Series, b: pd.Series, rtol: float = 1e-6, atol: float = 1e-8) -> tuple[bool, str]:
    """Whether two columns hold the same values row by row (numbers within a relative tolerance)."""
    if len(a) != len(b):
        return False, f"{len(a)} rows vs {len(b)}"
    a = a.reset_index(drop=True)
    b = b.reset_index(drop=True)
    num_a = pd.api.types.is_numeric_dtype(a) or pd.api.types.is_bool_dtype(a)
    num_b = pd.api.types.is_numeric_dtype(b) or pd.api.types.is_bool_dtype(b)
    if num_a and num_b:
        x = a.astype("float64").to_numpy(dtype=float, na_value=np.nan)
        y = b.astype("float64").to_numpy(dtype=float, na_value=np.nan)
        close = np.isclose(x, y, rtol=rtol, atol=atol, equal_nan=True)
        if close.all():
            return True, ""
        i = int(np.flatnonzero(~close)[0])
        return False, f"row {i}: {x[i]!r} vs {y[i]!r}"
    if pd.api.types.is_datetime64_any_dtype(a) or pd.api.types.is_datetime64_any_dtype(b):
        x = pd.to_datetime(a, errors="coerce", utc=True, format="mixed")
        y = pd.to_datetime(b, errors="coerce", utc=True, format="mixed")
        equal = (x == y) | (x.isna() & y.isna())
    else:
        x, y = a.astype(object), b.astype(object)
        equal = pd.Series([
            (pd.isna(u) and pd.isna(v)) if (_is_missing(u) or _is_missing(v)) else str(u) == str(v)
            for u, v in zip(x, y)
        ])
    if bool(equal.all()):
        return True, ""
    i = int(np.flatnonzero(~equal.to_numpy())[0])
    return False, f"row {i}: {a.iloc[i]!r} vs {b.iloc[i]!r}"


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _frames_match(out: pd.DataFrame, expected: pd.DataFrame, columns: list[str]) -> tuple[bool, str]:
    if len(out) != len(expected):
        return False, f"{len(out)} rows instead of {len(expected)}"
    for col in columns:
        if col not in out.columns:
            return False, f"column {col!r} is missing"
        ok, detail = same_values(out[col], expected[col])
        if not ok:
            return False, f"column {col!r} differs at {detail}"
    return True, ""


def run_ops(ops: list[tuple[int, Op]], frame: pd.DataFrame, live: bool, dropped: Optional[dict[int, int]] = None) -> pd.DataFrame:
    """Run the steps in order; ``dropped`` collects the step that removed each row id."""
    out = frame
    for index, op in ops:
        before = out[ROW_ID].to_numpy() if dropped is not None and op.is_filter and ROW_ID in out.columns else None
        try:
            out = op.apply(out, live)
        except Exception as exc:  # noqa: BLE001 - any failure is reported against its step
            raise StepError(index, type(op).__name__, exc) from exc
        if before is not None:
            for row in np.setdiff1d(before, out[ROW_ID].to_numpy()):
                dropped.setdefault(int(row), index)
    return out


def _batch_replay(step: StepReport, df: pd.DataFrame, params: dict[str, Any], op: Optional[Op]) -> Optional[pd.DataFrame]:
    """The step applied to the full training rows, or None when it cannot be reproduced here."""
    if op is not None:
        return op.apply(df, live=False)
    if not step.created and not step.modified and step.rows_before == step.rows_after:
        return df.drop(columns=[c for c in step.removed if c in df.columns])
    if step.action == "drop_duplicates":
        subset = params.get("columns") or _cols(df)
        keep = False if params.get("keep") == "none" else params.get("keep", "first")
        return df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
    if step.action == "sort_data":
        return df.sort_values(
            params["by"], ascending=params.get("ascending", True), na_position=params.get("na_position", "last"), kind="stable"
        ).reset_index(drop=True)
    if step.action == "drop_rows" and params.get("mode") == "row_range":
        n = len(df)
        end = min(params.get("end_row") if params.get("end_row") is not None else n, n)
        positions = np.zeros(n, dtype=bool)
        positions[int(params.get("start_row", 0)):end] = True
        drop = ~positions if params.get("keep_matching") else positions
        return df.loc[~drop].reset_index(drop=True)
    return None


def compile_lineage(
    chain: list[VersionInfo],
    *,
    feature_columns: list[str],
    target_column: str,
    time_column: Optional[str] = None,
) -> Compiled:
    """Rebuild, fit and check every step between the upload and the training version (the last in ``chain``)."""
    if not chain:
        raise ValueError("The model's dataset version has no history.")
    ctx = Context(target=target_column, time_column=time_column)
    steps: list[StepReport] = []
    ops: dict[int, Op] = {}
    blocking: list[str] = []
    warnings_: list[str] = []

    parent = chain[0].load()
    for index, info in enumerate(chain[1:], start=1):
        child = info.load()
        created, removed, modified, rows_changed = _diff(parent, child)
        step = StepReport(
            index=index, action=info.action, title=action_title(info.action),
            summary=_params_summary(info.action, info.params), version_id=info.id, mode=EXACT,
            created=created, removed=removed, modified=modified,
            rows_before=len(parent), rows_after=len(child),
        )
        action = info.action
        unchanged = not created and not modified and not rows_changed
        if unchanged and action not in _OPS:
            step.mode = NOT_NEEDED if removed else NO_CHANGE
            if removed:
                step.note = "Only removed columns, which the model does not use."
        elif action in _UPSTREAM_ACTIONS or (action == "combine_datasets" and info.params.get("mode") in ("merge", "join")):
            step.mode = UPSTREAM
            step.note = _UPSTREAM_ACTIONS.get(
                action, "Joining another dataset; the live feed must send the joined columns in each row."
            )
        elif action in _TRAINING_ONLY_NOTES or (action == "combine_datasets") or (
            action == "drop_rows" and info.params.get("mode") == "row_range"
        ):
            step.mode = TRAINING_ONLY
            step.note = _TRAINING_ONLY_NOTES.get(action, "Only shaped the training rows (appended rows or a fixed row range).")
        elif action in _OPS:
            try:
                op = _OPS[action](info.params, parent, child, ctx)
                ops[index] = op
                step.mode = op.mode
                step.note = op.note
                if unchanged:
                    step.note = " ".join(filter(None, [
                        "Changed nothing in the training data, but live rows can still need it.", op.note,
                    ]))
            except Unsupported as exc:
                step.mode = UNSUPPORTED
                step.note = str(exc)
            except Exception as exc:  # noqa: BLE001 - a step that cannot be rebuilt is reported, not fatal
                step.mode = UNSUPPORTED
                step.note = f"Could not rebuild this step: {exc}"
        else:
            step.mode = UNSUPPORTED
            step.note = "This step cannot run on live rows."
        if index in ops:
            try:
                ok, detail = _frames_match(ops[index].apply(parent.copy(), live=False), child, list(child.columns))
            except Exception as exc:  # noqa: BLE001
                ok, detail = False, f"replaying it failed: {exc}"
            step.verified = ok
            step.detail = "" if ok else detail
        steps.append(step)
        parent = child
    training = parent

    missing = [c for c in feature_columns if c not in training.columns]
    if missing:
        raise ValueError(f"The training version does not have the model's features: {missing}")

    # The live feed starts after the last step it must already have applied.
    cut = max((s.index for s in steps if s.mode == UPSTREAM), default=0)
    for s in steps:
        if s.index < cut and s.mode not in (NOT_NEEDED, NO_CHANGE, TRAINING_ONLY, UPSTREAM) and s.action not in (
            "rename_columns", "convert_types",
        ):
            warnings_.append(
                f"Step {s.index} ({s.title}) comes before step {cut} ({steps[cut - 1].title}), so the live feed "
                f"must send data that has already been through it."
            )
            s.mode = UPSTREAM
            s.note = f"Happens before step {cut}; the live feed must already include it."
        elif s.index < cut and s.mode not in (NOT_NEEDED, NO_CHANGE):
            s.mode = UPSTREAM
            s.note = s.note or f"Happens before step {cut}; the live feed must already include it."

    # Walk back from the model's features to the steps and raw columns they depend on.
    needed = set(feature_columns)
    for s in reversed(steps):
        if s.index <= cut:
            break
        op = ops.get(s.index)
        affects = set(s.created) | set(s.modified) | (op.writes() if op is not None else set())
        if s.mode in (UNSUPPORTED,) and (affects & needed):
            s.kept = True
            blocking.append(f"Step {s.index} ({s.title}): {s.note}")
            continue
        if op is None:
            continue
        if op.is_filter or affects & needed:
            if op.is_filter and not (affects & needed) and not op.requires(needed) and op.note:
                s.kept = False
                s.mode = TRAINING_ONLY
                continue
            s.kept = True
            needed = (needed - set(s.created)) | op.requires(needed)
        elif s.mode not in (NO_CHANGE,):
            s.mode = NOT_NEEDED
            s.note = s.note or "Nothing the model uses depends on this step."
    kept_ops = [(s.index, ops[s.index]) for s in steps if s.kept and s.index in ops]
    for s in steps:
        if s.kept and s.verified is False:
            blocking.append(
                f"Step {s.index} ({s.title}) did not reproduce its stored result when replayed ({s.detail})."
            )
        elif s.index > cut and s.mode == UNSUPPORTED and not s.kept and s.rows_before != s.rows_after:
            warnings_.append(
                f"Step {s.index} ({s.title}) removed training rows but cannot run live, so live rows are not filtered by it."
            )

    cut_version = chain[cut]
    cut_data = cut_version.load()
    if time_column not in cut_data.columns:
        time_column = detect_time_column(cut_data)
        ctx.time_column = time_column
    inputs = [c for c in cut_data.columns if c in needed]
    unknown = sorted(needed - set(cut_data.columns))
    if unknown:
        blocking.append(f"These columns are needed but not produced by any step the deployment can run: {unknown}")
    if time_column and time_column in cut_data.columns and time_column not in inputs:
        inputs.append(time_column)
    input_dtypes = {c: str(cut_data[c].dtype) for c in inputs}

    lookback = sum(op.lookback for _, op in kept_ops)
    buffer_rows = 0 if lookback == 0 and not any(op.mode in (WINDOW, APPROX) for _, op in kept_ops) else min(
        max(2 * lookback + 10, _MIN_BUFFER_ROWS), _MAX_BUFFER_ROWS
    )

    approx = _approximate_features(steps, ops, feature_columns)
    labels = {s.index: f"step {s.index} ({s.title})" for s in steps}
    compiled = Compiled(
        steps=steps, ops=kept_ops, inputs=inputs, input_dtypes=input_dtypes, time_column=time_column,
        buffer_rows=buffer_rows, cut_index=cut, cut_version_id=cut_version.id,
        blocking=blocking, warnings=warnings_, checks={}, approx_features=approx, labels=labels,
    )
    if not blocking:
        _check_end_to_end(compiled, chain, cut_data, training, feature_columns, ops)
    if approx:
        compiled.warnings.append(
            "These features are approximated on live rows because training used later readings: "
            + ", ".join(approx[:10]) + (" …" if len(approx) > 10 else "")
        )
    return compiled


def _approximate_features(steps: list[StepReport], ops: dict[int, Op], features: list[str]) -> list[str]:
    tainted: set[str] = set()
    for s in steps:
        if not s.kept or s.index not in ops:
            continue
        op = ops[s.index]
        outputs = set(s.created) | set(s.modified) | op.writes()
        if op.mode == APPROX or op.requires(outputs) & tainted or (set(s.modified) & tainted):
            tainted |= outputs
    return [f for f in features if f in tainted]


def _check_end_to_end(
    compiled: Compiled,
    chain: list[VersionInfo],
    cut_data: pd.DataFrame,
    training: pd.DataFrame,
    features: list[str],
    ops: dict[int, Op],
) -> None:
    checks = compiled.checks
    after_cut = [s for s in compiled.steps if s.index > compiled.cut_index]

    # 1. All steps on the full training rows must give the training version itself.
    batch: Optional[pd.DataFrame] = cut_data.copy()
    batch[ROW_ID] = np.arange(len(batch))
    blocker = None
    for s in after_cut:
        replayed = _batch_replay(s, batch, chain[s.index].params, ops.get(s.index))
        if replayed is None:
            blocker = s
            batch = None
            break
        batch = replayed
    if batch is not None:
        ok, detail = _frames_match(batch, training, features)
        checks["training_rows"] = {"ok": ok, "detail": detail}
        if not ok:
            compiled.blocking.append(f"Replaying every step on the training data did not reproduce the model's inputs ({detail}).")
            return
    else:
        checks["training_rows"] = {
            "ok": None,
            "detail": f"Skipped: step {blocker.index} ({blocker.title}) changes the training rows in a way that cannot be replayed; every step was checked on its own instead.",
        }

    # 2. Only the live inputs and only the kept steps, as a live feed would run them.
    live_frame = coerce_inputs(cut_data[compiled.inputs].copy(), compiled.input_dtypes)
    live_frame[ROW_ID] = np.arange(len(live_frame))
    try:
        live = run_ops(compiled.ops, live_frame, live=True)
    except StepError as exc:
        compiled.blocking.append(f"Running the steps on the live columns failed: {exc}")
        return
    exact = [f for f in features if f not in compiled.approx_features]
    if batch is not None:
        joined = batch[[ROW_ID, *features]].merge(live[[ROW_ID, *features]], on=ROW_ID, suffixes=("", "__live"))
        bad = []
        for f in exact:
            ok, detail = same_values(joined[f], joined[f"{f}__live"])
            if not ok:
                bad.append(f"{f} ({detail})")
        checks["live_columns"] = {"ok": not bad, "rows": int(len(joined)), "detail": "; ".join(bad[:5])}
        if bad:
            compiled.blocking.append("With only the live inputs, these features differ from training: " + "; ".join(bad[:5]))
            return

    # 3. Streaming: the newest rows arriving a few at a time with only the buffered history.
    n = len(live_frame)
    start = max(0, n - _STREAM_CHECK_ROWS)
    reference = live.set_index(ROW_ID)
    for attempt in range(4):
        buffer_rows = compiled.buffer_rows
        mismatched: list[str] = []
        compared = 0
        for chunk_start in range(start, n, _STREAM_CHUNK):
            chunk_end = min(chunk_start + _STREAM_CHUNK, n)
            window = live_frame.iloc[max(0, chunk_start - buffer_rows):chunk_end]
            try:
                out = run_ops(compiled.ops, window.copy(), live=True)
            except StepError as exc:
                compiled.blocking.append(f"Streaming rows through the steps failed: {exc}")
                return
            fresh = out[out[ROW_ID] >= chunk_start].set_index(ROW_ID)
            common = fresh.index.intersection(reference.index)
            compared += len(common)
            for f in exact:
                ok, detail = same_values(fresh.loc[common, f], reference.loc[common, f], rtol=1e-6, atol=1e-6)
                if not ok and f not in mismatched:
                    mismatched.append(f)
        if not mismatched or compiled.buffer_rows >= _MAX_BUFFER_ROWS or buffer_rows == 0 and not compiled.ops:
            break
        compiled.buffer_rows = min(max(compiled.buffer_rows * 2, 20), _MAX_BUFFER_ROWS)
    checks["streaming"] = {"ok": not mismatched, "rows": compared, "buffer_rows": compiled.buffer_rows,
                           "detail": ", ".join(mismatched[:5])}
    if mismatched:
        compiled.warnings.append(
            f"Streamed one at a time with {compiled.buffer_rows} rows of history, these features still differ "
            f"slightly from the batch result: {', '.join(mismatched[:5])}."
        )


# ---------------------------------------------------------------------------
# Live inputs
# ---------------------------------------------------------------------------


def coerce_inputs(frame: pd.DataFrame, dtypes: dict[str, str]) -> pd.DataFrame:
    """Live values in the types the columns had in the training data (bad values become missing)."""
    out = pd.DataFrame(index=frame.index)
    for col, dtype in dtypes.items():
        values = frame[col] if col in frame.columns else pd.Series([None] * len(frame), index=frame.index, dtype=object)
        if dtype.startswith(("int", "Int", "uint", "UInt", "float", "Float")):
            numbers = pd.to_numeric(values.map(lambda v: v.strip() if isinstance(v, str) else v), errors="coerce")
            numbers = numbers.astype("float64")
            if dtype.startswith(("int", "uint", "Int", "UInt")) and numbers.notna().all() and (numbers % 1 == 0).all():
                numbers = numbers.astype("int64")
            out[col] = numbers
        elif dtype in ("bool", "boolean"):
            flags = eda._to_boolean(values.astype(object))
            out[col] = flags.astype(bool) if flags.notna().all() else flags
        elif dtype.startswith("datetime64"):
            stamps = pd.to_datetime(values, errors="coerce", utc=True, format="mixed")
            if "UTC" not in dtype and "," not in dtype:
                stamps = stamps.dt.tz_localize(None)
            out[col] = stamps
        elif dtype == "category":
            out[col] = values.astype(object).astype("category")
        else:
            text = values.map(lambda v: v if v is None or isinstance(v, str) or _is_missing(v) else str(v))
            try:
                out[col] = text.astype(dtype)
            except (TypeError, ValueError):
                out[col] = text.astype(object)
    return out
