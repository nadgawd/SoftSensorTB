"""Manual mode: the tool forms of each pipeline step and running a tool by hand.

The forms are generated from the same JSON schemas the assistant is given, so
every parameter the assistant can set is a control in the UI, with the same
defaults, limits and validation. ``MANUAL_TOOLS`` only adds what a schema
cannot say: which step a tool belongs to, which fields pick columns, and
which fields matter for the option currently chosen.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, ValidationError, field_validator

from backend.agents.execution_agent import (
    _METRIC_TOOLS,
    _MUTATING_TOOLS,
    _PLOT_TOOLS,
    _TABLE_TOOLS,
    run_tool_manually,
)
from backend.auth import User, get_current_user
from backend.database import AsyncSessionLocal
from backend.limits import manual_limit_per_hour, manual_limiter
from backend.mcp_server import LLM_TOOL_SCHEMAS, TOOL_FUNCTIONS
from backend.mcp_server.eda_tools import _validate_uuid_str
from backend.mcp_server.safe_expr import FUNCTIONS as SAFE_FUNCTIONS
from backend.services.projects import owned_version

logger = logging.getLogger(__name__)

router = APIRouter(tags=["tools"])

STEPS = ("ingestion", "quality", "preprocessing", "feature-selection", "modeling", "validation", "export")

MULTI_COLUMN_FIELDS = frozenset({
    "columns", "y_columns", "feature_columns", "group_by", "value_columns", "id_columns", "by",
    "descending_columns", "path_columns", "on",
})
SINGLE_COLUMN_FIELDS = frozenset({
    "x_column", "z_column", "color_column", "target_column", "column", "stratify_column", "group_column",
    "pivot_column", "column_b", "value_column", "time_column", "condition_column",
})
CODE_FIELDS = frozenset({"sql_query", "formula", "query"})
# Widgets a JSON schema cannot express: another of the user's datasets, and a
# value → value table whose keys are typed rather than picked from the columns.
SPECIAL_WIDGETS = {"other_dataset": "dataset", "value_mapping": "value_mapping"}

_LABELS = {
    "sql_query": "SQL query",
    "n_neighbors": "Neighbours (k)",
    "max_iter": "Max iterations",
    "random_state": "Random seed",
    "flag_missing": "Add missing-value flag columns",
    "n_features_to_select": "Features to keep",
    "n_estimators": "Trees",
    "drop_anomalies": "Drop anomalous rows",
    "n_components": "Components",
    "keep_original": "Keep original columns",
    "n_bins": "Bins",
    "drop_first": "Drop first category",
    "interaction_only": "Interaction terms only",
    "k_neighbors": "Neighbours (k)",
    "x_column": "X column",
    "y_columns": "Y columns",
    "z_column": "Z column",
    "max_points": "Max points",
    "test_size": "Test fraction",
    "split_strategy": "Train/test split",
    "fit_intercept": "Fit intercept",
    "pls_scale": "Scale inside PLS",
    "knn_weights": "Neighbour weights",
    "model_id": "Model",
    "agg_func": "Aggregation",
    "min_periods": "Min periods",
    "center": "Centre the window",
    "lag_periods": "Lag periods",
    "new_column_name": "New column name",
    "column_mapping": "Rename columns",
    "feature_min": "Range min",
    "feature_max": "Range max",
    "quantile_low": "Lower percentile",
    "quantile_high": "Upper percentile",
    "lower_bound": "Lower bound",
    "upper_bound": "Upper bound",
    "category_order": "Category order (low → high)",
    "stratify_column": "Stratify by",
    "n_strata": "Strata (quantile bins)",
    "cv_strategy": "Cross-validation",
    "cv_folds": "Folds (k)",
    "cv_shuffle": "Shuffle before folding",
    "view": "View",
    "n_rows": "Rows to show",
    "top_n": "Values per column",
    "start_row": "From row (0-based)",
    "end_row": "To row (exclusive)",
    "keep_matching": "Keep only matching rows (filter)",
    "target_type": "Convert to",
    "errors": "Unconvertible values",
    "datetime_format": "Date format",
    "replace_with": "Replace with",
    "use_regex": "Regular expression",
    "score_func": "Score function",
    "k": "Features to keep (k)",
    "validation_size": "Validation fraction",
    "group_column": "Group column",
    "cv_repeats": "Repeats",
    "cv_test_size": "Validation fraction per split",
    "cv_window": "Training window (rows)",
    "n_bootstrap": "Bootstrap resamples",
    "interpolation_method": "Interpolation method",
    "interpolation_order": "Spline / polynomial order",
    "limit_direction": "Fill direction",
    "time_column": "Time column",
    "window_type": "Window type",
    "span": "Span (EWM)",
    "positive_class": "Event class",
    "pair_operation": "Pair operation",
    "holiday_country": "Holiday country (ISO code)",
    "perplexity": "Perplexity (t-SNE)",
    "hidden_units": "Hidden units",
    "group_by": "Group by",
    "pivot_column": "Pivot column",
    "value_columns": "Value columns",
    "normalize": "Normalize (crosstab)",
    "margins": "Add totals",
    "column_b": "Second column",
    "mu": "Hypothesised mean (μ)",
    "equal_var": "Assume equal variances",
    "adf_regression": "ADF regression",
    "by": "Sort by",
    "descending_columns": "Descending columns",
    "na_position": "Missing values",
    "id_columns": "ID columns",
    "var_name": "Variable column name",
    "value_name": "Value column name",
    "other_dataset": "Other dataset",
    "on": "Key columns",
    "join_type": "Join type",
    "add_source_column": "Add source column",
    "condition_column": "Condition column",
    "value_mapping": "Value mapping",
    "unmapped": "Unmapped values",
    "in_place": "Replace the columns",
    "frequency": "New time step",
    "fill": "Fill new time steps with",
    "fill_limit": "Max steps filled in a row",
    "drop_empty": "Drop empty intervals",
    "n_features": "Features to keep",
    "n_repeats": "Permutation repeats",
    "value_column": "Value column",
    "path_columns": "Path / hierarchy columns",
    "n_clusters": "Clusters (k)",
    "positive_when": "Event when target is",
    "n_points": "Training sizes",
}

_STRATIFIED_CV = ["stratified_kfold", "repeated_stratified_kfold", "stratified_shuffle_split"]
_STRATIFIED = [{"split_strategy": ["stratified"]}, {"cv_strategy": _STRATIFIED_CV}]
_RANDOM_CV = ["repeated_kfold", "repeated_stratified_kfold", "shuffle_split", "stratified_shuffle_split", "bootstrap"]

_NAMES_RULE = "Refer to a column by its name. Names with spaces or symbols go in backticks or square brackets: `Feed temp` or [Feed temp]."
_FORMULA_HELP = {
    "syntax": [
        _NAMES_RULE,
        "Arithmetic: +  −  *  /  ** (power)  // (whole division)  % (remainder); use ( ) to group.",
        "Functions: " + ", ".join(SAFE_FUNCTIONS) + " (log is the natural log).",
        "Comparisons (>  <  >=  <=  ==  !=) give True/False; combine them with & (and), | (or), ~ (not).",
        "Not allowed: other functions, methods such as .mean(), @variables and indexing.",
    ],
    "examples": [
        ["[Unnamed: 8] * [Unnamed: 8]", "square of a column whose name has a space and a colon"],
        ["(`Feed temp` - 273.15) * flow", "temperature in °C times flow"],
        ["log(pressure) + sqrt(abs(dP))", "math functions of columns"],
        ["(temp > 350) & (flow < 10)", "True where both hold"],
    ],
}
_QUERY_HELP = {
    "syntax": [
        _NAMES_RULE,
        "Compare with >  <  >=  <=  ==  !=; text values go in quotes: status == 'ok'.",
        "Combine with and / or / not (or & | ~); a list of values: unit in ['A', 'B'].",
        "Arithmetic and the formula functions work too: abs(error) > 3.",
        "Rows where the query is true are dropped (or, with 'Keep only the matching rows', kept).",
    ],
    "examples": [
        ["temp > 350 and status == 'ok'", "hot rows marked ok"],
        ["[flow rate] <= 0 or [flow rate] > 500", "flow outside 0–500"],
        ["unit in ['A', 'C']", "rows of units A or C"],
    ],
}
_SQL_HELP = {
    "syntax": [
        "DuckDB SQL over the active dataset, which is the table named dataset.",
        'Quote names with spaces or symbols in double quotes: "Feed temp".',
        "Read-only: SELECT or WITH queries (WHERE, GROUP BY, ORDER BY, LIMIT, window functions…), DESCRIBE and SUMMARIZE.",
    ],
    "examples": [
        ["SELECT * FROM dataset LIMIT 10", "first 10 rows"],
        ['SELECT unit, avg("Feed temp") AS mean_temp FROM dataset GROUP BY unit', "mean per unit"],
        ["SELECT * FROM dataset WHERE flow > 5 ORDER BY ts DESC", "filter and sort"],
    ],
}

_FREQUENCY_HELP = {
    "syntax": [
        "A number and a unit: s (seconds), min (minutes), h (hours), D (days), W (weeks), MS (months).",
        "Downsample: a step longer than the spacing of the rows, e.g. 1-min data to 1h.",
        "Upsample: a step shorter than the spacing, e.g. 4-hourly lab samples to 15min.",
        "Intervals start on round times (00:00, 01:00…) and are labelled by their start; W weeks end on Sunday.",
    ],
    "examples": [
        ["1h", "hourly"],
        ["15min", "every 15 minutes"],
        ["30s", "every 30 seconds"],
        ["1D", "daily"],
        ["1MS", "monthly"],
    ],
}

_TRAIN_SHOW_WHEN = {
    "n_components": {"algorithm": ["PLS", "PCR"]},
    "pls_scale": {"algorithm": ["PLS"]},
    "alpha": {"algorithm": ["RIDGE", "LASSO"]},
    "max_iter": {"algorithm": ["LASSO"]},
    "fit_intercept": {"algorithm": ["OLS", "RIDGE", "LASSO"]},
    "n_neighbors": {"algorithm": ["KNN"]},
    "knn_weights": {"algorithm": ["KNN"]},
    "random_state": [
        {"split_strategy": ["random", "stratified", "group"]},
        {"cv_strategy": ["kfold", "stratified_kfold", *_RANDOM_CV]},
    ],
    "stratify_column": _STRATIFIED,
    "n_strata": _STRATIFIED,
    "group_column": [{"split_strategy": ["group"]}, {"cv_strategy": ["group_kfold", "leave_one_group_out"]}],
    "cv_folds": {"cv_strategy": [
        "kfold", "stratified_kfold", "time_series", "group_kfold", "repeated_kfold",
        "repeated_stratified_kfold", "shuffle_split", "stratified_shuffle_split", "walk_forward",
    ]},
    "cv_shuffle": {"cv_strategy": ["kfold", "stratified_kfold"]},
    "cv_repeats": {"cv_strategy": ["repeated_kfold", "repeated_stratified_kfold"]},
    "cv_test_size": {"cv_strategy": ["shuffle_split", "stratified_shuffle_split"]},
    "cv_window": {"cv_strategy": ["walk_forward"]},
    "n_bootstrap": {"cv_strategy": ["bootstrap"]},
}

# Which generate_custom_plot fields each plot type reads.
_PLOT_FIELDS: dict[str, tuple[str, ...]] = {
    "x_column": (
        "scatter", "time_series", "histogram", "box_plot", "violin_plot", "scatter_3d", "bar", "count", "swarm",
        "strip", "point", "lollipop", "pie", "donut", "joint", "hexbin", "bubble", "contour", "histogram_2d",
        "line", "area", "candlestick", "waterfall",
    ),
    "color_column": (
        "scatter", "time_series", "histogram", "box_plot", "violin_plot", "scatter_3d", "pair_plot", "kde",
        "ridgeline", "ecdf", "bar", "count", "strip", "point", "joint", "bubble", "contour",
        "parallel_coordinates", "radviz", "andrews_curves", "network", "dendrogram", "chord", "silhouette",
    ),
    "trendline": ("scatter",),
    "z_column": ("scatter_3d",),
    "value_column": (
        "pie", "donut", "bubble", "histogram_2d", "treemap", "sunburst", "sankey", "network", "chord",
    ),
    "path_columns": ("treemap", "sunburst", "sankey", "network", "chord"),
    "agg_func": ("bar", "lollipop", "pie", "donut", "histogram_2d"),
    "bins": ("hexbin", "contour", "histogram_2d", "candlestick"),
    "lags": ("acf", "pacf"),
    "n_clusters": ("silhouette",),
    "threshold": ("network", "chord"),
}
_PLOT_SHOW_WHEN = {field: {"plot_type": list(types)} for field, types in _PLOT_FIELDS.items()}

_STAT_TESTS_ONE_COLUMN = ["t_test_one_sample", "t_test_independent", "anova", "chi_square", "shapiro", "kolmogorov_smirnov", "adf"]

_VALUE_OPERATORS = [
    "equals", "not_equals", "greater_than", "greater_or_equal", "less_than", "less_or_equal", "contains",
]
_REPLACE_ONLY = {"operation": ["replace"]}
_PATTERN_OPERATIONS = ["replace", "contains", "extract"]
_PROJECTIONS = ["pca", "truncated_svd", "lda", "tsne", "autoencoder"]
_COLUMN_FUNCTIONS = [
    "cumsum", "cumprod", "cummax", "cummin", "rank", "pct_rank", "zscore", "minmax", "demean",
    "group_mean", "group_sum", "group_count", "group_min", "group_max", "group_std", "group_median",
]

_DIAGNOSTIC = {"steps": ("validation",), "kind": "plot"}

# label, steps (the one pipeline step whose Tools list shows the tool; empty
# for a tool with its own panel, named by "panel"), and optional: numeric (column pickers limited to numeric
# columns), show_when ({field: {other_field: [values]}}, all of which must
# hold, or a list of such dicts of which any may hold), defaults (form
# defaults for required fields), app_defaults (fill from the workbench:
# "target" or "features"), advanced (fields folded under "Advanced").
MANUAL_TOOLS: dict[str, dict[str, Any]] = {
    "inspect_data": {
        "label": "Inspect data (head, info, describe…)",
        "steps": ("ingestion",),
        "defaults": {"view": "head"},
        "show_when": {
            "n_rows": {"view": ["head", "tail"]},
            "top_n": {"view": ["value_counts"]},
            "method": {"view": ["correlation"]},
            "quantiles": {"view": ["distribution"]},
        },
    },
    "aggregate_data": {
        "label": "Group by / pivot table / crosstab",
        "steps": ("ingestion",),
        "show_when": {
            "pivot_column": {"mode": ["pivot_table", "crosstab"]},
            "normalize": {"mode": ["crosstab"]},
            "margins": {"mode": ["pivot_table", "crosstab"]},
        },
    },
    "statistical_test": {
        "label": "Statistical tests (t-test, ANOVA, χ², normality, ADF)",
        "steps": ("quality",),
        "defaults": {"test": "shapiro"},
        "show_when": {
            "column": {"test": ["t_test_one_sample", "t_test_independent", "t_test_paired", "anova", "chi_square"]},
            "column_b": {"test": ["t_test_paired", "chi_square"]},
            "group_column": {"test": ["t_test_independent", "anova"]},
            "columns": {"test": ["shapiro", "kolmogorov_smirnov", "adf"]},
            "mu": {"test": ["t_test_one_sample"]},
            "equal_var": {"test": ["t_test_independent"]},
            "adf_regression": {"test": ["adf"]},
        },
    },
    "list_datasets": {
        "label": "Your uploaded datasets",
        "steps": ("ingestion",),
    },
    "sort_data": {
        "label": "Sort rows",
        "steps": ("ingestion",),
    },
    "combine_datasets": {
        "label": "Combine datasets (merge, join, concat)",
        "steps": ("ingestion",),
        "show_when": {
            "on": {"mode": ["merge"]},
            "suffix": {"mode": ["merge", "join"]},
            "add_source_column": {"mode": ["concat"]},
        },
    },
    "reshape_data": {
        "label": "Reshape (melt, stack, pivot, unstack)",
        "steps": ("preprocessing",),
        "defaults": {"mode": "melt"},
        "show_when": {
            "pivot_column": {"mode": ["pivot", "unstack"]},
            "var_name": {"mode": ["melt", "stack", "unstack"]},
            "value_name": {"mode": ["melt", "stack", "unstack"]},
            "agg_func": {"mode": ["pivot"]},
        },
    },
    "get_column_statistics": {
        "label": "Column statistics",
        "steps": ("ingestion",),
    },
    "run_duckdb_query": {
        "label": "SQL query",
        "steps": ("export",),
        "defaults": {"sql_query": "SELECT * FROM dataset LIMIT 10"},
        "help": {"sql_query": _SQL_HELP},
    },
    "convert_types": {
        "label": "Convert types / parse dates",
        "steps": ("ingestion",),
        "defaults": {"target_type": "float"},
        "show_when": {"datetime_format": {"target_type": ["datetime"]}},
    },
    "clean_text": {
        "label": "Text & regex (strip, case, replace, contains, extract)",
        "steps": ("quality",),
        "defaults": {"operation": "strip"},
        "show_when": {
            "find": {"operation": _PATTERN_OPERATIONS},
            "replace_with": _REPLACE_ONLY,
            "use_regex": {"operation": ["replace", "contains"]},
            "ignore_case": {"operation": _PATTERN_OPERATIONS},
        },
    },
    "drop_rows": {
        "label": "Drop / filter rows (condition, range, query)",
        "steps": ("quality",),
        "defaults": {"mode": "condition"},
        "show_when": {
            "query": {"mode": ["query"]},
            "column": {"mode": ["condition"]},
            "operator": {"mode": ["condition"]},
            "value": {"mode": ["condition"], "operator": _VALUE_OPERATORS},
            "start_row": {"mode": ["row_range"]},
            "end_row": {"mode": ["row_range"]},
        },
        "help": {"query": _QUERY_HELP},
    },
    "mask_values": {
        "label": "Replace where a condition holds (mask / where)",
        "steps": ("quality",),
        "show_when": {"value": {"operator": _VALUE_OPERATORS}},
    },
    "map_values": {
        "label": "Map / recode values",
        "steps": ("quality",),
    },
    "drop_duplicates": {
        "label": "Drop duplicates",
        "steps": ("quality",),
    },
    "remove_missing_data": {
        "label": "Missing values",
        "steps": ("quality",),
        "defaults": {"strategy": "drop_rows"},
        "show_when": {
            "n_neighbors": {"strategy": ["knn"]},
            "max_iter": {"strategy": ["iterative"]},
            "random_state": {"strategy": ["iterative"]},
            "interpolation_method": {"strategy": ["interpolate"]},
            "interpolation_order": {"strategy": ["interpolate"], "interpolation_method": ["spline", "polynomial"]},
            "limit_direction": {"strategy": ["interpolate"]},
            "time_column": {"strategy": ["interpolate"], "interpolation_method": ["time"]},
        },
        "advanced": ("max_iter", "random_state"),
    },
    "remove_outliers": {
        "label": "Outliers (drop, cap, clip)",
        "steps": ("quality",),
        "numeric": ("columns",),
        "defaults": {"method": "iqr"},
        "show_when": {
            "threshold": {"method": ["z_score", "iqr", "winsorize"]},
            "action": {"method": ["z_score", "iqr"]},
            "lower_bound": {"method": ["clip"]},
            "upper_bound": {"method": ["clip"]},
        },
    },
    "detect_anomalies": {
        "label": "Anomaly detection (Isolation Forest)",
        "steps": ("quality",),
        "numeric": ("columns",),
        "advanced": ("n_estimators", "random_state"),
    },
    "drop_columns": {
        "label": "Drop columns",
        "steps": ("feature-selection",),
    },
    "rename_columns": {
        "label": "Rename columns",
        "steps": ("ingestion",),
    },
    "normalize_data": {
        "label": "Normalize / scale",
        "steps": ("preprocessing",),
        "numeric": ("columns",),
        "defaults": {"strategy": "standard"},
        "show_when": {
            "feature_min": {"strategy": ["min_max"]},
            "feature_max": {"strategy": ["min_max"]},
            "quantile_low": {"strategy": ["robust"]},
            "quantile_high": {"strategy": ["robust"]},
        },
    },
    "transform_features": {
        "label": "Transform (log, Box-Cox, bins…)",
        "steps": ("preprocessing",),
        "numeric": ("columns",),
        "defaults": {"strategy": "log"},
        "show_when": {
            "n_bins": {"strategy": ["bin_equal_width", "bin_equal_freq"]},
            "threshold": {"strategy": ["binarize"]},
        },
    },
    "encode_categorical": {
        "label": "Encode categorical (one-hot, target, LOO, WoE…)",
        "steps": ("preprocessing",),
        "defaults": {"strategy": "one_hot"},
        "app_defaults": {"target_column": "target"},
        "show_when": {
            "target_column": {"strategy": ["target", "leave_one_out", "woe"]},
            "positive_class": {"strategy": ["woe"]},
            "smoothing": {"strategy": ["target"]},
            "drop_first": {"strategy": ["one_hot"]},
            "category_order": {"strategy": ["ordinal"]},
        },
    },
    "rolling_aggregate": {
        "label": "Window functions (rolling, expanding, EWM)",
        "steps": ("preprocessing",),
        "numeric": ("columns",),
        "defaults": {"window": 5},
        "show_when": {
            "window": {"window_type": ["rolling"]},
            "span": {"window_type": ["ewm"]},
            "min_periods": {"window_type": ["rolling", "expanding"]},
            "center": {"window_type": ["rolling"]},
        },
        "advanced": ("min_periods", "center"),
    },
    "create_lag_features": {
        "label": "Lag / lead / diff (shift, diff, pct_change)",
        "steps": ("preprocessing",),
        "defaults": {"lag_periods": [1, 2, 3]},
    },
    "apply_function": {
        "label": "Apply function (map / transform)",
        "steps": ("preprocessing",),
        "numeric": ("columns",),
        "defaults": {"function": "zscore"},
        "show_when": {
            "decimals": {"function": ["round"]},
            "group_by": {"function": _COLUMN_FUNCTIONS},
        },
    },
    "feature_engineer": {
        "label": "Feature engineering (polynomial, dates, pairs…)",
        "steps": ("preprocessing",),
        "defaults": {"strategy": "polynomial"},
        "show_when": {
            "degree": {"strategy": ["polynomial"]},
            "interaction_only": {"strategy": ["polynomial"]},
            "period": {"strategy": ["cyclical"]},
            "pair_operation": {"strategy": ["pairwise"]},
            "holiday_country": {"strategy": ["datetime"]},
        },
    },
    "resample_time_series": {
        "label": "Resample time series (downsample / upsample)",
        "steps": ("preprocessing",),
        "defaults": {"frequency": "1h", "mode": "downsample"},
        "show_when": {
            "aggregation": {"mode": ["downsample"]},
            "drop_empty": {"mode": ["downsample"]},
            "fill": {"mode": ["upsample"]},
            "fill_limit": {"mode": ["upsample"], "fill": ["interpolate", "ffill", "bfill", "nearest"]},
        },
        "help": {"frequency": _FREQUENCY_HELP},
    },
    "execute_formula": {
        "label": "Formula column",
        "steps": ("preprocessing",),
        "help": {"formula": _FORMULA_HELP},
    },
    "balance_data": {
        "label": "Balance classes (SMOTE, ADASYN, over/undersampling)",
        "steps": ("preprocessing",),
        "defaults": {"strategy": "smote"},
        "show_when": {"k_neighbors": {"strategy": ["smote", "adasyn"]}},
        "advanced": ("random_state",),
    },
    "generate_custom_plot": {
        "label": "Plot",
        "steps": (),
        "panel": "graph",
        "defaults": {"plot_type": "correlation_heatmap"},
        "show_when": _PLOT_SHOW_WHEN,
        "advanced": ("max_points",),
    },
    "select_features_rfe": {
        "label": "Recursive feature elimination",
        "steps": ("feature-selection",),
        "numeric": ("target_column", "feature_columns"),
        "app_defaults": {"target_column": "target"},
        "show_when": {"alpha": {"estimator": ["ridge", "lasso"]}},
        "advanced": ("step",),
    },
    "select_k_best": {
        "label": "SelectKBest",
        "steps": ("feature-selection",),
        "numeric": ("target_column", "feature_columns"),
        "app_defaults": {"target_column": "target"},
        "show_when": {"random_state": {"score_func": ["mutual_info_regression"]}},
        "advanced": ("random_state",),
    },
    "drop_collinear_features": {
        "label": "Drop collinear features",
        "steps": ("feature-selection",),
        "numeric": ("target_column",),
        "app_defaults": {"target_column": "target"},
    },
    "reduce_dimensions": {
        "label": "Dimensionality reduction",
        "steps": ("feature-selection",),
        "numeric": ("columns",),
        "defaults": {"strategy": "pca"},
        "show_when": {
            "n_components": {"strategy": _PROJECTIONS},
            "keep_original": {"strategy": _PROJECTIONS},
            "threshold": {"strategy": ["variance_threshold"]},
            "target_column": {"strategy": ["lda"]},
            "perplexity": {"strategy": ["tsne"]},
            "hidden_units": {"strategy": ["autoencoder"]},
            "max_iter": {"strategy": ["tsne", "autoencoder"]},
            "random_state": {"strategy": ["truncated_svd", "tsne", "autoencoder"]},
        },
        "advanced": ("max_iter", "random_state"),
    },
    "select_features_model": {
        "label": "Model-based selection (importance, permutation, SHAP)",
        "steps": ("feature-selection",),
        "numeric": ("target_column", "feature_columns"),
        "app_defaults": {"target_column": "target"},
        "show_when": {
            "n_estimators": {"method": ["tree_importance", "permutation"]},
            "n_repeats": {"method": ["permutation"]},
            "test_size": {"method": ["permutation"]},
        },
        "advanced": ("n_estimators", "alpha", "random_state"),
    },
    "train_soft_sensor": {
        "label": "1 · Split & train a model",
        "steps": ("modeling",),
        "numeric": ("target_column", "feature_columns"),
        "defaults": {"algorithm": "PLS"},
        "app_defaults": {"target_column": "target", "feature_columns": "features"},
        "show_when": _TRAIN_SHOW_WHEN,
    },
    "evaluate_on_test_set": {
        "label": "2 · Test the model on the held-out test rows",
        "steps": ("modeling",),
    },
    "view_data_split": {
        "label": "View the train / validation / test rows",
        "steps": ("modeling",),
        "defaults": {"subset": "summary"},
        "show_when": {"n_rows": {"subset": ["train", "validation", "test"]}},
    },
    "list_trained_models": {
        "label": "Trained models",
        "steps": ("export",),
    },
    "generate_parity_plot": {"label": "Parity plot", **_DIAGNOSTIC},
    "generate_residuals_plot": {"label": "Residuals plot", **_DIAGNOSTIC},
    "generate_importance_plot": {"label": "Feature importance", **_DIAGNOSTIC},
    "generate_coefficients_plot": {"label": "Coefficients plot", **_DIAGNOSTIC},
    "generate_roc_curve": {"label": "ROC curve", **_DIAGNOSTIC},
    "generate_pr_curve": {"label": "Precision-recall curve", **_DIAGNOSTIC},
    "generate_learning_curve": {"label": "Learning curve", **_DIAGNOSTIC, "advanced": ("random_state",)},
}


def _kind(name: str) -> str:
    if name == "evaluate_on_test_set":
        return "test"
    if name in _METRIC_TOOLS:
        return "train"
    if name in _MUTATING_TOOLS:
        return "transform"
    if name in _PLOT_TOOLS:
        return "plot"
    if name in _TABLE_TOOLS:
        return "table"
    return "run"


def _deref(node: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
    if "$ref" in node:
        target = defs[node["$ref"].rsplit("/", 1)[-1]]
        return {**target, **{k: v for k, v in node.items() if k != "$ref"}}
    if "allOf" in node and len(node["allOf"]) == 1:
        merged = {**_deref(node["allOf"][0], defs), **{k: v for k, v in node.items() if k != "allOf"}}
        return merged
    return node


def _unwrap_nullable(node: dict[str, Any], defs: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """``anyOf: [X, null]`` → (X with the outer default/description, True)."""
    options = node.get("anyOf")
    if not options:
        return _deref(node, defs), False
    concrete = [o for o in options if o.get("type") != "null"]
    nullable = len(concrete) < len(options)
    if len(concrete) != 1:
        raise ValueError(f"Unsupported schema union: {node}")
    outer = {k: v for k, v in node.items() if k != "anyOf"}
    return {**_deref(concrete[0], defs), **outer}, nullable


def _label(name: str) -> str:
    return _LABELS.get(name) or name.replace("_", " ").capitalize()


def _widget(name: str, node: dict[str, Any]) -> str:
    if name in SPECIAL_WIDGETS:
        return SPECIAL_WIDGETS[name]
    if "enum" in node:
        return "select"
    kind = node.get("type")
    if kind == "boolean":
        return "checkbox"
    if kind in ("integer", "number"):
        return kind
    if kind == "array":
        items = node.get("items") or {}
        if items.get("type") == "integer":
            return "int_list"
        return "columns" if name in MULTI_COLUMN_FIELDS else "text_list"
    if kind == "object":
        return "mapping"
    if name in SINGLE_COLUMN_FIELDS:
        return "column"
    if name == "model_id":
        return "model"
    return "code" if name in CODE_FIELDS else "text"


def _clean_description(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _field_spec(tool: str, name: str, raw: dict[str, Any], required: bool, defs: dict[str, Any]) -> dict[str, Any]:
    ui = MANUAL_TOOLS[tool]
    node, nullable = _unwrap_nullable(raw, defs)
    spec: dict[str, Any] = {
        "name": name,
        "label": _label(name),
        "widget": _widget(name, node),
        "description": _clean_description(node.get("description", "")),
        "required": required,
        "nullable": nullable,
        "advanced": name in ui.get("advanced", ()),
    }
    if name in ui.get("help", {}):
        spec["help"] = ui["help"][name]
    if "enum" in node:
        spec["options"] = list(node["enum"])
    for key, out in (
        ("minimum", "min"), ("maximum", "max"),
        ("exclusiveMinimum", "exclusive_min"), ("exclusiveMaximum", "exclusive_max"),
        ("minItems", "min_items"),
    ):
        if key in node:
            spec[out] = node[key]
    if "default" in node:
        spec["default"] = node["default"]
    if name in ui.get("defaults", {}):
        spec["default"] = ui["defaults"][name]
    if name in ui.get("numeric", ()):
        spec["numeric_only"] = True
    if name in ui.get("app_defaults", {}):
        spec["app_default"] = ui["app_defaults"][name]
    if name in ui.get("show_when", {}):
        spec["show_when"] = ui["show_when"][name]
    return spec


def _schemas_by_name() -> dict[str, dict[str, Any]]:
    return {s["function"]["name"]: s["function"] for s in LLM_TOOL_SCHEMAS}


def manual_tool_specs() -> list[dict[str, Any]]:
    schemas = _schemas_by_name()
    out = []
    for name, ui in MANUAL_TOOLS.items():
        fn = schemas[name]
        params = fn["parameters"]
        defs = params.get("$defs", {})
        required = set(params.get("required", []))
        out.append({
            "name": name,
            "label": ui["label"],
            "description": _clean_description(fn["description"]),
            "steps": list(ui["steps"]),
            "panel": ui.get("panel"),
            "kind": _kind(name),
            "fields": [
                _field_spec(name, field, raw, field in required, defs)
                for field, raw in params.get("properties", {}).items()
            ],
        })
    return out


@router.get("/api/tools/manual")
async def get_manual_tools() -> dict[str, Any]:
    return {"steps": list(STEPS), "tools": manual_tool_specs()}


class ManualRunRequest(BaseModel):
    dataset_version_id: str = Field(..., description="Active dataset version UUID.")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Tool arguments, as in the tool schema.")

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        return _validate_uuid_str(value)


def _validation_message(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ()) if p != "dataset_version_id")
        msg = str(err.get("msg", "invalid value")).removeprefix("Value error, ")
        parts.append(f"{loc}: {msg}" if loc else msg)
    return "; ".join(parts) or "Invalid arguments."


@router.post("/api/tools/{tool_name}/run")
async def run_manual_tool(
    tool_name: str,
    request: ManualRunRequest,
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Run one tool with the arguments the user set in the step's form."""
    if tool_name not in MANUAL_TOOLS or tool_name not in TOOL_FUNCTIONS:
        raise HTTPException(status_code=404, detail=f"Unknown tool: {tool_name}")
    manual_limiter.check(user.id, manual_limit_per_hour(), "Manual run limit reached ({limit} runs per hour)")
    async with AsyncSessionLocal() as session:
        await owned_version(session, request.dataset_version_id, user)
    arguments: dict[str, Any] = {k: v for k, v in request.arguments.items() if v is not None}
    try:
        return await run_tool_manually(tool_name, arguments, request.dataset_version_id)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=_validation_message(exc)) from exc
    except TypeError as exc:
        # An argument the tool does not take.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 — tool errors (bad column, singular fit…) go back to the form
        logger.info("Manual %s failed: %s", tool_name, exc)
        raise HTTPException(status_code=400, detail=str(exc)[:500] or type(exc).__name__) from exc


def tools_for_step(step: str) -> list[str]:
    return [name for name, ui in MANUAL_TOOLS.items() if step in ui["steps"]]


__all__ = ["MANUAL_TOOLS", "STEPS", "manual_tool_specs", "router", "tools_for_step"]
