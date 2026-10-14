"""Read-only analysis tools: aggregation tables and statistical tests.

Both return ``{dataset_version_id, title, rows}`` tables like ``inspect_data``
and never create a dataset version.
"""

from __future__ import annotations

import math
import warnings
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import AsyncSessionLocal
from backend.mcp_server.eda_tools import (
    _assert_columns_exist,
    _assert_numeric_columns,
    _clean_names,
    _clean_optional_names,
    _get_version,
    _is_plain_numeric,
    _load_dataframe,
    _num,
    _openai_tool_schema,
    _uniform_rows,
    _validate_uuid_str,
)

_MAX_TABLE_ROWS = 500
_MAX_PIVOT_COLUMNS = 50
_MAX_TEST_COLUMNS = 30
_MAX_GROUPS = 50
_SHAPIRO_LIMIT = 5000


def _blank(value: Any) -> Any:
    return value.strip() or None if isinstance(value, str) else value


# ---------------------------------------------------------------------------
# groupby / pivot_table / crosstab
# ---------------------------------------------------------------------------


class AggregateMode(str, Enum):
    groupby = "groupby"
    pivot_table = "pivot_table"
    crosstab = "crosstab"


class AggFunc(str, Enum):
    count = "count"
    sum = "sum"
    mean = "mean"
    median = "median"
    min = "min"
    max = "max"
    std = "std"
    var = "var"
    nunique = "nunique"
    first = "first"
    last = "last"


_NUMERIC_AGGS = frozenset({AggFunc.sum, AggFunc.mean, AggFunc.median, AggFunc.std, AggFunc.var})


class CrosstabNormalize(str, Enum):
    none = "none"
    all = "all"
    index = "index"
    columns = "columns"


class AggregateDataArgs(BaseModel):
    dataset_version_id: str = Field(..., min_length=36, max_length=36)
    mode: AggregateMode = Field(
        AggregateMode.groupby,
        description=(
            "groupby: one row per combination of group_by with agg_func of each value column "
            "(df.groupby().agg()); pivot_table: group_by as rows, the values of pivot_column as "
            "columns, agg_func of one value column in the cells; crosstab: counts (or agg_func of a "
            "value column) for every pair of group_by and pivot_column values (pd.crosstab)."
        ),
    )
    group_by: list[str] = Field(..., min_length=1, description="Columns whose values form the groups / table rows.")
    pivot_column: Optional[str] = Field(None, description="For pivot_table and crosstab: the column whose values become the table columns.")
    value_columns: Optional[list[str]] = Field(
        None,
        description=(
            "Columns aggregated. groupby: omit to use every numeric column; pivot_table: one "
            "column (omit to count rows); crosstab: at most one column (omit to count rows)."
        ),
    )
    agg_func: AggFunc = Field(AggFunc.mean, description="Aggregation: count, sum, mean, median, min, max, std, var, nunique, first, last.")
    normalize: CrosstabNormalize = Field(
        CrosstabNormalize.none,
        description="For crosstab counts: show shares of the whole table (all), of each row (index) or of each column (columns).",
    )
    margins: bool = Field(False, description="For pivot_table and crosstab: add an 'All' row and column with totals.")

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        return _validate_uuid_str(value)

    @field_validator("group_by")
    @classmethod
    def _names(cls, value: list[str]) -> list[str]:
        return _clean_names(value)

    @field_validator("value_columns")
    @classmethod
    def _optional(cls, value: Optional[list[str]]) -> Optional[list[str]]:
        return _clean_optional_names(value)

    @field_validator("pivot_column", mode="before")
    @classmethod
    def _blank_pivot(cls, value: Any) -> Any:
        return _blank(value)

    @model_validator(mode="after")
    def _mode_needs(self) -> "AggregateDataArgs":
        if self.mode != AggregateMode.groupby:
            if not self.pivot_column:
                raise ValueError(f"{self.mode.value} needs pivot_column")
            if self.pivot_column in self.group_by:
                raise ValueError("pivot_column cannot also be in group_by")
            if self.value_columns and len(self.value_columns) > 1:
                raise ValueError(f"{self.mode.value} aggregates one value column")
        if self.normalize != CrosstabNormalize.none and self.mode != AggregateMode.crosstab:
            raise ValueError("normalize applies to crosstab only")
        overlap = set(self.value_columns or []) & set(self.group_by + ([self.pivot_column] if self.pivot_column else []))
        if overlap:
            raise ValueError(f"value_columns cannot also be grouping columns: {sorted(overlap)}")
        return self


def _flat(label: Any) -> str:
    if isinstance(label, tuple):
        return " | ".join(str(part) for part in label if str(part) != "")
    return str(label)


def _table_rows(table: pd.DataFrame) -> list[dict[str, Any]]:
    table = table.copy()
    table.columns = [_flat(c) for c in table.columns]
    return table.head(_MAX_TABLE_ROWS).to_dict(orient="records")


def _check_values(df: pd.DataFrame, values: list[str], func: AggFunc) -> None:
    if func in _NUMERIC_AGGS:
        _assert_numeric_columns(df, values)
    else:
        _assert_columns_exist(df, values)


def _aggregate(df: pd.DataFrame, args: AggregateDataArgs) -> tuple[str, list[dict[str, Any]], int]:
    keys = args.group_by
    _assert_columns_exist(df, keys)
    func = args.agg_func.value
    if args.mode == AggregateMode.groupby:
        values = args.value_columns or [c for c in df.columns if c not in keys and _is_plain_numeric(df[c])]
        _check_values(df, values, args.agg_func)
        grouped = df.groupby(keys, dropna=False, observed=True, sort=True)
        table = grouped.size().rename("rows").to_frame()
        if values:
            aggregated = grouped[values].agg(func)
            aggregated.columns = [f"{c}_{func}" for c in values]
            table = table.join(aggregated)
        table = table.reset_index()
        title = f"groupby {', '.join(keys)} — {func} of {', '.join(values) if values else 'row counts'}"
        return title, _table_rows(table), len(table)

    pivot = args.pivot_column
    _assert_columns_exist(df, [pivot])
    distinct = int(df[pivot].nunique(dropna=True))
    if distinct > _MAX_PIVOT_COLUMNS:
        raise ValueError(f"pivot_column {pivot!r} has {distinct} distinct values; at most {_MAX_PIVOT_COLUMNS} fit as columns")
    value = (args.value_columns or [None])[0]
    if value:
        _check_values(df, [value], args.agg_func)
    if args.mode == AggregateMode.pivot_table:
        if value:
            table = pd.pivot_table(
                df, index=keys, columns=pivot, values=value, aggfunc=func,
                margins=args.margins, observed=True,
            )
            described = f"{func} of {value}"
        else:
            table = pd.crosstab([df[k] for k in keys], df[pivot], margins=args.margins)
            described = "row counts"
        table = table.reset_index()
        title = f"pivot_table — {described}; rows {', '.join(keys)}, columns {pivot}"
        return title, _table_rows(table), len(table)

    normalize: Any = {"none": False, "all": "all", "index": "index", "columns": "columns"}[args.normalize.value]
    if value:
        table = pd.crosstab(
            [df[k] for k in keys], df[pivot], values=df[value], aggfunc=func,
            normalize=normalize, margins=args.margins,
        )
        described = f"{func} of {value}"
    else:
        table = pd.crosstab([df[k] for k in keys], df[pivot], normalize=normalize, margins=args.margins)
        described = "counts" if normalize is False else f"shares (normalized over {args.normalize.value})"
    table = table.reset_index()
    title = f"crosstab — {described}; rows {', '.join(keys)}, columns {pivot}"
    return title, _table_rows(table), len(table)


async def aggregate_data(
    dataset_version_id: str,
    group_by: list[str],
    mode: str = "groupby",
    pivot_column: Optional[str] = None,
    value_columns: Optional[list[str]] = None,
    agg_func: str = "mean",
    normalize: str = "none",
    margins: bool = False,
    *,
    session: Optional[AsyncSession] = None,
) -> dict[str, Any]:
    """Summarise groups of rows (groupby, pivot_table, crosstab); never modifies the data."""
    args = AggregateDataArgs(
        dataset_version_id=dataset_version_id,
        mode=mode,  # type: ignore[arg-type]
        group_by=group_by,
        pivot_column=pivot_column,
        value_columns=value_columns,
        agg_func=agg_func,  # type: ignore[arg-type]
        normalize=normalize,  # type: ignore[arg-type]
        margins=margins,
    )

    async def _run(db: AsyncSession) -> dict[str, Any]:
        version = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(version)
        title, rows, total = _aggregate(df, args)
        if total > _MAX_TABLE_ROWS:
            title += f" (first {_MAX_TABLE_ROWS} of {total} rows)"
        return {"dataset_version_id": args.dataset_version_id, "title": title, "rows": _uniform_rows(rows)}

    if session is not None:
        return await _run(session)
    async with AsyncSessionLocal() as db:
        return await _run(db)


# ---------------------------------------------------------------------------
# Hypothesis, normality and stationarity tests
# ---------------------------------------------------------------------------


class StatTest(str, Enum):
    t_test_one_sample = "t_test_one_sample"
    t_test_independent = "t_test_independent"
    t_test_paired = "t_test_paired"
    anova = "anova"
    chi_square = "chi_square"
    shapiro = "shapiro"
    kolmogorov_smirnov = "kolmogorov_smirnov"
    adf = "adf"


_MULTI_COLUMN_TESTS = frozenset({StatTest.shapiro, StatTest.kolmogorov_smirnov, StatTest.adf})
_TEST_NAMES = {
    StatTest.t_test_one_sample: "One-sample t-test",
    StatTest.t_test_independent: "Independent two-sample t-test",
    StatTest.t_test_paired: "Paired t-test",
    StatTest.anova: "One-way ANOVA",
    StatTest.chi_square: "Chi-square test of independence",
    StatTest.shapiro: "Shapiro-Wilk normality test",
    StatTest.kolmogorov_smirnov: "Kolmogorov-Smirnov normality test (Lilliefors)",
    StatTest.adf: "Augmented Dickey-Fuller stationarity test",
}


class AdfRegression(str, Enum):
    c = "c"
    ct = "ct"
    n = "n"


class StatisticalTestArgs(BaseModel):
    dataset_version_id: str = Field(..., min_length=36, max_length=36)
    test: StatTest = Field(
        ...,
        description=(
            "t_test_one_sample (mean of column vs mu), t_test_independent (column between the 2 "
            "groups of group_column; Welch unless equal_var), t_test_paired (column vs column_b on "
            "the same rows), anova (column across the groups of group_column), chi_square "
            "(independence of the categorical column and column_b), shapiro (Shapiro-Wilk), "
            "kolmogorov_smirnov (KS against a normal with the sample's mean and std, Lilliefors "
            "p-value) or adf (augmented Dickey-Fuller unit-root test of a time series in row order)."
        ),
    )
    column: Optional[str] = Field(
        None,
        description="The numeric column tested (t-tests, anova) or the first categorical column (chi_square).",
    )
    column_b: Optional[str] = Field(None, description="Second column: paired t-test (numeric) or chi_square (categorical).")
    group_column: Optional[str] = Field(None, description="Groups compared by t_test_independent (exactly 2) and anova (2 or more).")
    columns: Optional[list[str]] = Field(
        None,
        description=f"For shapiro, kolmogorov_smirnov and adf: the numeric columns tested (omit for every numeric column, at most {_MAX_TEST_COLUMNS}).",
    )
    mu: float = Field(0.0, description="For t_test_one_sample: the hypothesised mean.")
    equal_var: bool = Field(False, description="For t_test_independent: assume equal variances (Student) instead of Welch's t-test.")
    alpha: float = Field(0.05, description="Significance level for the conclusion.", gt=0, lt=0.5)
    adf_regression: AdfRegression = Field(
        AdfRegression.c,
        description="For adf: deterministic terms — c (constant), ct (constant and trend) or n (none).",
    )

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        return _validate_uuid_str(value)

    @field_validator("column", "column_b", "group_column", mode="before")
    @classmethod
    def _blank_names(cls, value: Any) -> Any:
        return _blank(value)

    @field_validator("columns")
    @classmethod
    def _optional(cls, value: Optional[list[str]]) -> Optional[list[str]]:
        return _clean_optional_names(value)

    @model_validator(mode="after")
    def _test_needs(self) -> "StatisticalTestArgs":
        test = self.test
        if test in _MULTI_COLUMN_TESTS:
            if self.columns and len(self.columns) > _MAX_TEST_COLUMNS:
                raise ValueError(f"Test at most {_MAX_TEST_COLUMNS} columns at once")
            return self
        if not self.column:
            raise ValueError(f"{test.value} needs column")
        if test in (StatTest.t_test_paired, StatTest.chi_square):
            if not self.column_b:
                raise ValueError(f"{test.value} needs column_b")
            if self.column_b == self.column:
                raise ValueError("column_b must differ from column")
        if test in (StatTest.t_test_independent, StatTest.anova):
            if not self.group_column:
                raise ValueError(f"{test.value} needs group_column")
            if self.group_column == self.column:
                raise ValueError("group_column must differ from column")
        return self


def _verdict(p: Optional[float], alpha: float, reject: str, keep: str) -> str:
    if p is None:
        return "undetermined"
    return f"{reject} (p < {alpha:g})" if p < alpha else f"{keep} (p ≥ {alpha:g})"


def _numeric_values(df: pd.DataFrame, column: str) -> pd.Series:
    _assert_numeric_columns(df, [column])
    return df[column].dropna().astype(float)


def _check_spread(values: pd.Series, what: str) -> None:
    if values.nunique() < 2:
        raise ValueError(f"{what} has no variation (all values are equal); the test is undefined")


def _group_values(df: pd.DataFrame, column: str, group_column: str) -> list[tuple[str, np.ndarray]]:
    _assert_numeric_columns(df, [column])
    _assert_columns_exist(df, [group_column])
    rows = df[[column, group_column]].dropna()
    groups = [(str(label), part[column].astype(float).to_numpy()) for label, part in rows.groupby(group_column, sort=True)]
    if len(groups) > _MAX_GROUPS:
        raise ValueError(f"group_column {group_column!r} has {len(groups)} groups; at most {_MAX_GROUPS}")
    return groups


def _two_sample(df: pd.DataFrame, args: StatisticalTestArgs) -> dict[str, Any]:
    from scipy import stats

    test = args.test
    if test == StatTest.t_test_one_sample:
        x = _numeric_values(df, args.column)
        if len(x) < 2:
            raise ValueError(f"{args.column!r} needs at least 2 values")
        _check_spread(x, repr(args.column))
        r = stats.ttest_1samp(x, args.mu)
        return {
            "column": args.column, "n": len(x), "statistic": r.statistic, "p_value": r.pvalue, "df": r.df,
            "mean": x.mean(), "mu": args.mu,
            "result": _verdict(r.pvalue, args.alpha, f"the mean differs from {args.mu:g}", f"no evidence the mean differs from {args.mu:g}"),
        }
    if test == StatTest.t_test_paired:
        _assert_numeric_columns(df, [args.column, args.column_b])
        pairs = df[[args.column, args.column_b]].dropna().astype(float)
        if len(pairs) < 2:
            raise ValueError("The paired t-test needs at least 2 rows with both values")
        diff = pairs[args.column] - pairs[args.column_b]
        _check_spread(diff, "The difference between the columns")
        r = stats.ttest_rel(pairs[args.column], pairs[args.column_b])
        return {
            "column": f"{args.column} − {args.column_b}", "n": len(pairs), "statistic": r.statistic,
            "p_value": r.pvalue, "df": r.df, "mean_difference": diff.mean(),
            "result": _verdict(r.pvalue, args.alpha, "the paired means differ", "no evidence the paired means differ"),
        }
    groups = _group_values(df, args.column, args.group_column)
    if test == StatTest.t_test_independent:
        if len(groups) != 2:
            raise ValueError(
                f"The independent t-test compares exactly 2 groups; {args.group_column!r} has {len(groups)}. "
                "Use anova for more."
            )
        (la, a), (lb, b) = groups
        if len(a) < 2 or len(b) < 2:
            raise ValueError("Each group needs at least 2 values")
        r = stats.ttest_ind(a, b, equal_var=args.equal_var)
        return {
            "column": args.column, "groups": f"{la} vs {lb}", "n": len(a) + len(b),
            "statistic": r.statistic, "p_value": r.pvalue, "df": r.df,
            f"mean_{la}": float(np.mean(a)), f"mean_{lb}": float(np.mean(b)),
            "variant": "Student (equal variances)" if args.equal_var else "Welch (unequal variances)",
            "result": _verdict(r.pvalue, args.alpha, "the group means differ", "no evidence the group means differ"),
        }
    if len(groups) < 2:
        raise ValueError(f"anova needs at least 2 groups in {args.group_column!r}")
    if any(len(values) < 2 for _, values in groups):
        small = [label for label, values in groups if len(values) < 2]
        raise ValueError(f"Every group needs at least 2 values; too small: {small[:10]}")
    r = stats.f_oneway(*[values for _, values in groups])
    everything = np.concatenate([values for _, values in groups])
    grand = everything.mean()
    between = sum(len(v) * (v.mean() - grand) ** 2 for _, v in groups)
    total = float(((everything - grand) ** 2).sum())
    return {
        "column": args.column, "groups": len(groups), "n": len(everything), "statistic": r.statistic,
        "p_value": r.pvalue, "df_between": len(groups) - 1, "df_within": len(everything) - len(groups),
        "eta_squared": between / total if total > 0 else None,
        "result": _verdict(r.pvalue, args.alpha, "at least one group mean differs", "no evidence the group means differ"),
    }


def _chi_square(df: pd.DataFrame, args: StatisticalTestArgs) -> dict[str, Any]:
    from scipy import stats

    _assert_columns_exist(df, [args.column, args.column_b])
    table = pd.crosstab(df[args.column], df[args.column_b])
    if table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError("chi_square needs at least 2 values in each column")
    if table.shape[0] > _MAX_GROUPS or table.shape[1] > _MAX_GROUPS:
        raise ValueError(f"chi_square supports at most {_MAX_GROUPS} values per column")
    r = stats.chi2_contingency(table.to_numpy())
    n = int(table.to_numpy().sum())
    cramers_v = math.sqrt(r.statistic / (n * (min(table.shape) - 1))) if n else None
    low_expected = float((r.expected_freq < 5).mean())
    row = {
        "column": f"{args.column} × {args.column_b}", "n": n, "statistic": r.statistic, "p_value": r.pvalue,
        "df": r.dof, "cramers_v": cramers_v,
        "result": _verdict(r.pvalue, args.alpha, "the columns are associated", "no evidence of association"),
    }
    if low_expected > 0.2:
        row["warning"] = f"{100 * low_expected:.0f}% of cells expect fewer than 5 rows; the p-value may be unreliable"
    return row


def _per_column(df: pd.DataFrame, args: StatisticalTestArgs) -> list[dict[str, Any]]:
    from scipy import stats

    if args.columns:
        _assert_numeric_columns(df, args.columns)
        columns = args.columns
    else:
        columns = [c for c in df.columns if _is_plain_numeric(df[c])][:_MAX_TEST_COLUMNS]
    if not columns:
        raise ValueError("No numeric columns to test")
    rows: list[dict[str, Any]] = []
    for c in columns:
        x = df[c].dropna().astype(float)
        row: dict[str, Any] = {"column": c, "n": len(x)}
        minimum = 10 if args.test == StatTest.adf else 3
        if len(x) < minimum or x.nunique() < 2:
            row["result"] = f"skipped: needs at least {minimum} values that are not all equal"
            rows.append(row)
            continue
        if args.test == StatTest.shapiro:
            sample = x.sample(n=_SHAPIRO_LIMIT, random_state=42) if len(x) > _SHAPIRO_LIMIT else x
            r = stats.shapiro(sample)
            row.update(statistic=r.statistic, p_value=r.pvalue)
            if len(x) > _SHAPIRO_LIMIT:
                row["note"] = f"tested a random sample of {_SHAPIRO_LIMIT} values"
            row["result"] = _verdict(r.pvalue, args.alpha, "not normal", "consistent with a normal distribution")
        elif args.test == StatTest.kolmogorov_smirnov:
            from statsmodels.stats.diagnostic import lilliefors

            statistic, p = lilliefors(x.to_numpy(), dist="norm", pvalmethod="table")
            row.update(statistic=statistic, p_value=p)
            row["result"] = _verdict(float(p), args.alpha, "not normal", "consistent with a normal distribution")
        else:
            from statsmodels.tools.sm_exceptions import SingularMatrixWarning
            from statsmodels.tsa.stattools import adfuller

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", SingularMatrixWarning)
                statistic, p, used_lag, nobs, critical, _icbest = adfuller(
                    x.to_numpy(), autolag="AIC", regression=args.adf_regression.value, result_object=False,
                )
            row.update(
                statistic=statistic, p_value=p, lags_used=used_lag, observations=nobs,
                **{f"critical_{k}": v for k, v in critical.items()},
            )
            row["result"] = _verdict(float(p), args.alpha, "stationary (unit root rejected)", "non-stationary (unit root not rejected)")
            if any(issubclass(w.category, SingularMatrixWarning) for w in caught):
                row["note"] = "the regression was rank-deficient (a deterministic or exactly periodic series); the result is unreliable"
        rows.append(row)
    return rows


async def statistical_test(
    dataset_version_id: str,
    test: str,
    column: Optional[str] = None,
    column_b: Optional[str] = None,
    group_column: Optional[str] = None,
    columns: Optional[list[str]] = None,
    mu: float = 0.0,
    equal_var: bool = False,
    alpha: float = 0.05,
    adf_regression: str = "c",
    *,
    session: Optional[AsyncSession] = None,
) -> dict[str, Any]:
    """Hypothesis (t-tests, ANOVA, chi-square), normality (Shapiro-Wilk, KS) and stationarity (ADF) tests."""
    args = StatisticalTestArgs(
        dataset_version_id=dataset_version_id,
        test=test,  # type: ignore[arg-type]
        column=column,
        column_b=column_b,
        group_column=group_column,
        columns=columns,
        mu=mu,
        equal_var=equal_var,
        alpha=alpha,
        adf_regression=adf_regression,  # type: ignore[arg-type]
    )

    async def _run(db: AsyncSession) -> dict[str, Any]:
        version = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(version)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if args.test in _MULTI_COLUMN_TESTS:
                rows = _per_column(df, args)
            elif args.test == StatTest.chi_square:
                rows = [_chi_square(df, args)]
            else:
                rows = [_two_sample(df, args)]
        for row in rows:
            for key in ("statistic", "p_value", "mean", "mean_difference", "eta_squared", "cramers_v"):
                if key in row:
                    row[key] = _num(row[key], 6)
        rows = [{"test": _TEST_NAMES[args.test], **row} for row in rows]
        return {
            "dataset_version_id": args.dataset_version_id,
            "title": f"{_TEST_NAMES[args.test]} — α = {args.alpha:g}",
            "rows": _uniform_rows(rows),
        }

    if session is not None:
        return await _run(session)
    async with AsyncSessionLocal() as db:
        return await _run(db)


LLM_TOOL_SCHEMAS: list[dict[str, Any]] = [
    _openai_tool_schema(
        name="aggregate_data",
        description=(
            "Summarise groups of rows without changing the data: groupby with an aggregation "
            "(count, sum, mean, median, min, max, std, var, nunique, first, last), pivot_table, "
            "or crosstab (frequency table, optionally normalized, with margins)."
        ),
        args_model=AggregateDataArgs,
    ),
    _openai_tool_schema(
        name="statistical_test",
        description=(
            "Run a statistical test and report statistic, p-value and conclusion: t-tests "
            "(one-sample, independent/Welch, paired), one-way ANOVA, chi-square independence, "
            "normality (Shapiro-Wilk, Kolmogorov-Smirnov/Lilliefors) or stationarity (augmented "
            "Dickey-Fuller). Read-only."
        ),
        args_model=StatisticalTestArgs,
    ),
]

TOOL_FUNCTIONS = {
    "aggregate_data": aggregate_data,
    "statistical_test": statistical_test,
}

__all__ = ["LLM_TOOL_SCHEMAS", "TOOL_FUNCTIONS", "aggregate_data", "statistical_test"]
