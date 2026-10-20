"""Plotly builders for the generate_custom_plot types beyond the basic ones.

Every builder takes the (already subsampled) DataFrame and the validated
``GenerateCustomPlotArgs`` and returns a ``go.Figure``. They raise
``ValueError`` with a message the user can act on when columns are missing or
of the wrong kind.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

DISTRIBUTION = ("kde", "rug", "ridgeline", "ecdf", "qq_plot")
CATEGORICAL = ("bar", "count", "swarm", "strip", "point", "lollipop", "pie", "donut")
MULTIVARIATE = (
    "joint", "hexbin", "bubble", "contour", "histogram_2d",
    "parallel_coordinates", "radviz", "andrews_curves",
)
SEQUENTIAL = ("line", "area", "candlestick", "waterfall", "acf", "pacf")
RELATIONAL = ("treemap", "sunburst", "sankey", "network", "dendrogram", "chord")
CLUSTERING = ("silhouette",)
EXTRA_PLOT_TYPES = DISTRIBUTION + CATEGORICAL + MULTIVARIATE + SEQUENTIAL + RELATIONAL + CLUSTERING

_MAX_CATEGORIES = 50
_MAX_GROUPS = 12
_MAX_DEFAULT_COLUMNS = 8
_MAX_CURVES = 200
_MAX_NODES = 150
_PALETTE = px.colors.qualitative.Plotly
_MISSING = "(missing)"


# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------


def _exists(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in dataset: {missing}")


def _is_number(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _numeric(df: pd.DataFrame, columns: Sequence[str]) -> None:
    _exists(df, columns)
    bad = [c for c in columns if not _is_number(df[c])]
    if bad:
        raise ValueError(f"Columns must be numeric: {bad}")


def _numeric_columns(df: pd.DataFrame, exclude: Sequence[str] = ()) -> list[str]:
    return [c for c in df.columns if c not in exclude and _is_number(df[c])]


def _value_columns(args: Any, kind: str, *, minimum: int = 1) -> list[str]:
    columns = list(args.y_columns) or ([args.x_column] if args.x_column else [])
    if len(columns) < minimum:
        raise ValueError(f"{kind} needs {'a column' if minimum == 1 else f'at least {minimum} columns'} in y_columns")
    return columns


def _default_numeric(df: pd.DataFrame, args: Any, kind: str, minimum: int, limit: int = _MAX_DEFAULT_COLUMNS) -> list[str]:
    exclude = [args.color_column] if args.color_column else []
    columns = list(args.y_columns) or _numeric_columns(df, exclude)[:limit]
    if len(columns) < minimum:
        raise ValueError(f"{kind} needs at least {minimum} numeric columns in y_columns")
    _numeric(df, columns)
    return columns


def _labels(series: pd.Series) -> pd.Series:
    """Category labels as text, with missing values shown as (missing)."""
    return series.astype(object).where(series.notna(), _MISSING).astype(str)


def _need(value: Optional[str], kind: str, field: str) -> str:
    if not value:
        raise ValueError(f"{kind} needs {field}")
    return value


def _groups(df: pd.DataFrame, column: Optional[str]) -> list[tuple[str, pd.DataFrame]]:
    """(label, rows) per value of ``column`` (most frequent first), or one group of all rows."""
    if not column:
        return [("", df)]
    _exists(df, [column])
    labels = _labels(df[column])
    order = labels.value_counts().index[:_MAX_GROUPS]
    return [(str(label), df[labels == label]) for label in order]


_AGGREGATIONS = {"count", "sum", "mean", "median", "min", "max"}


def _aggregate(df: pd.DataFrame, by: list[str], value: Optional[str], how: str) -> pd.DataFrame:
    """One row per combination of ``by`` with ``value`` aggregated (or the row count) in column ``value``."""
    frame = df[by].apply(_labels)
    if value and how != "count":
        _numeric(df, [value])
        frame["value"] = df[value].astype(float)
        out = frame.groupby(by, sort=False)["value"].agg(how).reset_index()
    else:
        out = frame.groupby(by, sort=False).size().reset_index(name="value")
    return out


def _top_categories(table: pd.DataFrame, column: str, limit: int = _MAX_CATEGORIES) -> pd.DataFrame:
    totals = table.groupby(column)["value"].sum().abs().sort_values(ascending=False)
    keep = set(totals.index[:limit])
    return table[table[column].isin(keep)]


def _value_label(args: Any, value: Optional[str]) -> str:
    return f"{args.agg_func}({value})" if value and args.agg_func != "count" else "count"


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------


def _kde_curve(values: np.ndarray, grid: np.ndarray) -> Optional[np.ndarray]:
    from scipy.stats import gaussian_kde

    if len(values) < 2 or np.ptp(values) == 0:
        return None
    return gaussian_kde(values)(grid)


def _series_sets(df: pd.DataFrame, args: Any, kind: str) -> list[tuple[str, np.ndarray]]:
    """(name, finite values) per column, or per color group when there is one column and a color_column."""
    columns = _value_columns(args, kind)
    _numeric(df, columns)
    if args.color_column and len(columns) == 1:
        return [
            (label, rows[columns[0]].dropna().astype(float).to_numpy())
            for label, rows in _groups(df, args.color_column)
        ]
    return [(c, df[c].dropna().astype(float).to_numpy()) for c in columns]


def _kde(df: pd.DataFrame, args: Any) -> go.Figure:
    sets = [(name, v) for name, v in _series_sets(df, args, "kde") if len(v)]
    if not sets:
        raise ValueError("kde has no values to plot")
    low = min(float(v.min()) for _, v in sets)
    high = max(float(v.max()) for _, v in sets)
    pad = (high - low) * 0.1 or 1.0
    grid = np.linspace(low - pad, high + pad, 256)
    fig = go.Figure()
    for i, (name, values) in enumerate(sets):
        density = _kde_curve(values, grid)
        if density is None:
            continue
        fig.add_trace(go.Scatter(
            x=grid, y=density, mode="lines", name=name, fill="tozeroy", opacity=0.6,
            line={"color": _PALETTE[i % len(_PALETTE)]},
        ))
    if not fig.data:
        raise ValueError("kde needs at least 2 different values per series")
    fig.update_layout(title="Density (KDE)", xaxis_title="value", yaxis_title="density")
    return fig


def _rug(df: pd.DataFrame, args: Any) -> go.Figure:
    columns = _value_columns(args, "rug")
    _numeric(df, columns)
    fig = go.Figure()
    for i, c in enumerate(columns):
        values = df[c].dropna().astype(float)
        fig.add_trace(go.Scatter(
            x=values, y=[c] * len(values), mode="markers", name=c,
            marker={"symbol": "line-ns-open", "size": 16, "color": _PALETTE[i % len(_PALETTE)]},
            hovertemplate="%{x}<extra>" + c + "</extra>",
        ))
    fig.update_layout(title="Rug plot", xaxis_title="value", showlegend=False)
    return fig


def _ridgeline(df: pd.DataFrame, args: Any) -> go.Figure:
    sets = [(name or "all rows", v) for name, v in _series_sets(df, args, "ridgeline") if len(v)]
    if not sets:
        raise ValueError("ridgeline has no values to plot")
    fig = go.Figure()
    for i, (name, values) in enumerate(sets):
        fig.add_trace(go.Violin(
            x=values, name=name, side="positive", orientation="h", width=2.5,
            points=False, line={"width": 1}, meanline={"visible": True},
            fillcolor=_PALETTE[i % len(_PALETTE)], opacity=0.7, line_color="#333",
        ))
    fig.update_layout(title="Ridgeline plot", xaxis_title="value", violinmode="overlay", violingap=0, showlegend=False)
    return fig


def _ecdf(df: pd.DataFrame, args: Any) -> go.Figure:
    columns = _value_columns(args, "ecdf")
    _numeric(df, columns)
    if len(columns) == 1:
        if args.color_column:
            _exists(df, [args.color_column])
        frame = df[[columns[0]] + ([args.color_column] if args.color_column else [])].dropna(subset=[columns[0]])
        return px.ecdf(frame, x=columns[0], color=args.color_column, title=f"ECDF of {columns[0]}")
    melted = df[columns].melt(var_name="column", value_name="value").dropna()
    return px.ecdf(melted, x="value", color="column", title="ECDF")


def _qq(df: pd.DataFrame, args: Any) -> go.Figure:
    from scipy import stats

    columns = _value_columns(args, "qq_plot")
    _numeric(df, columns)
    fig = go.Figure()
    for i, c in enumerate(columns):
        values = df[c].dropna().astype(float).to_numpy()
        if len(values) < 3:
            raise ValueError(f"qq_plot needs at least 3 values in {c!r}")
        (theoretical, ordered), (slope, intercept, _r) = stats.probplot(values, dist="norm")
        color = _PALETTE[i % len(_PALETTE)]
        fig.add_trace(go.Scatter(x=theoretical, y=ordered, mode="markers", name=c, marker={"color": color, "size": 5}))
        ends = np.array([theoretical.min(), theoretical.max()])
        fig.add_trace(go.Scatter(
            x=ends, y=slope * ends + intercept, mode="lines", name=f"{c} normal line",
            line={"color": color, "dash": "dash"},
        ))
    title = f"Normal Q-Q plot of {columns[0]}" if len(columns) == 1 else "Normal Q-Q plot"
    fig.update_layout(title=title, xaxis_title="theoretical quantiles", yaxis_title="sample quantiles")
    return fig


# ---------------------------------------------------------------------------
# Categorical comparisons
# ---------------------------------------------------------------------------


def _bar(df: pd.DataFrame, args: Any) -> go.Figure:
    x = _need(args.x_column, "bar", "x_column (the categories)")
    value = args.y_columns[0] if args.y_columns else None
    by = [x] + ([args.color_column] if args.color_column and args.color_column != x else [])
    _exists(df, by)
    table = _top_categories(_aggregate(df, by, value, args.agg_func), x)
    table = table.sort_values("value", ascending=False)
    label = _value_label(args, value)
    fig = px.bar(
        table, x=x, y="value", color=by[1] if len(by) > 1 else None, barmode="group",
        labels={"value": label}, title=f"{label} by {x}",
    )
    fig.update_xaxes(type="category")
    return fig


def _count(df: pd.DataFrame, args: Any) -> go.Figure:
    x = _need(args.x_column, "count", "x_column (the categories)")
    by = [x] + ([args.color_column] if args.color_column and args.color_column != x else [])
    _exists(df, by)
    table = _top_categories(_aggregate(df, by, None, "count"), x).sort_values("value", ascending=False)
    fig = px.bar(
        table, x=x, y="value", color=by[1] if len(by) > 1 else None, barmode="group",
        labels={"value": "count"}, title=f"Count of rows per {x}",
    )
    fig.update_xaxes(type="category")
    return fig


def _beeswarm_offsets(values: np.ndarray, span: float, width: float = 0.8) -> np.ndarray:
    """Horizontal offsets that spread points sharing a value band side by side."""
    if len(values) == 0:
        return np.array([])
    band = span / 60 if span > 0 else 1.0
    bins = np.floor((values - values.min()) / band).astype(int)
    offsets = np.zeros(len(values))
    largest = int(np.bincount(bins).max())
    step = width / max(largest, 1)
    for b in np.unique(bins):
        members = np.flatnonzero(bins == b)
        rank = np.arange(len(members))
        offsets[members] = np.where(rank % 2 == 0, rank // 2, -(rank // 2 + 1)) * step
        offsets[members] += step / 2 if len(members) % 2 == 0 else 0
    return offsets


def _swarm(df: pd.DataFrame, args: Any) -> go.Figure:
    if not args.y_columns:
        raise ValueError("swarm needs a numeric column in y_columns")
    y = args.y_columns[0]
    _numeric(df, [y])
    values_all = df[y].dropna().astype(float)
    span = float(values_all.max() - values_all.min()) if len(values_all) else 0.0
    groups = _groups(df, args.x_column)
    fig = go.Figure()
    for position, (label, rows) in enumerate(groups):
        values = rows[y].dropna().astype(float).to_numpy()[:2000]
        offsets = _beeswarm_offsets(values, span)
        fig.add_trace(go.Scatter(
            x=position + offsets, y=values, mode="markers", name=label or y,
            marker={"size": 5, "color": _PALETTE[position % len(_PALETTE)]},
            hovertemplate=f"{label or y}: " + "%{y}<extra></extra>",
        ))
    fig.update_layout(
        title=f"Swarm plot of {y}" + (f" by {args.x_column}" if args.x_column else ""),
        yaxis_title=y, showlegend=False,
        xaxis={"tickmode": "array", "tickvals": list(range(len(groups))), "ticktext": [g[0] or y for g in groups]},
    )
    return fig


def _strip(df: pd.DataFrame, args: Any) -> go.Figure:
    if not args.y_columns:
        raise ValueError("strip needs a numeric column in y_columns")
    y = args.y_columns[0]
    _numeric(df, [y])
    cols = [y] + [c for c in (args.x_column, args.color_column) if c]
    _exists(df, cols)
    frame = df[list(dict.fromkeys(cols))].copy()
    for c in (args.x_column, args.color_column):
        if c:
            frame[c] = _labels(frame[c])
    return px.strip(
        frame, x=args.x_column, y=y, color=args.color_column,
        title=f"Strip plot of {y}" + (f" by {args.x_column}" if args.x_column else ""),
    )


def _point(df: pd.DataFrame, args: Any) -> go.Figure:
    from scipy import stats

    x = _need(args.x_column, "point", "x_column (the categories)")
    if not args.y_columns:
        raise ValueError("point needs a numeric column in y_columns")
    _numeric(df, args.y_columns)
    _exists(df, [x])
    series_sets: list[tuple[str, str, pd.DataFrame]] = []
    if args.color_column and len(args.y_columns) == 1:
        series_sets = [(label, args.y_columns[0], rows) for label, rows in _groups(df, args.color_column)]
    else:
        series_sets = [(c, c, df) for c in args.y_columns]
    order = list(_labels(df[x]).value_counts().index[:_MAX_CATEGORIES])
    fig = go.Figure()
    for i, (name, column, rows) in enumerate(series_sets):
        labels = _labels(rows[x])
        means, errors = [], []
        for category in order:
            values = rows.loc[labels == category, column].dropna().astype(float)
            means.append(float(values.mean()) if len(values) else None)
            if len(values) > 1:
                errors.append(float(stats.t.ppf(0.975, len(values) - 1) * values.std() / math.sqrt(len(values))))
            else:
                errors.append(0.0)
        fig.add_trace(go.Scatter(
            x=order, y=means, mode="lines+markers", name=name,
            error_y={"type": "data", "array": errors, "visible": True},
            line={"color": _PALETTE[i % len(_PALETTE)]},
        ))
    fig.update_layout(title=f"Mean with 95% confidence interval by {x}", xaxis_title=x, yaxis_title="mean")
    fig.update_xaxes(type="category")
    return fig


def _lollipop(df: pd.DataFrame, args: Any) -> go.Figure:
    x = _need(args.x_column, "lollipop", "x_column (the categories)")
    value = args.y_columns[0] if args.y_columns else None
    _exists(df, [x])
    table = _top_categories(_aggregate(df, [x], value, args.agg_func), x).sort_values("value")
    label = _value_label(args, value)
    stems_x: list[Any] = []
    stems_y: list[Any] = []
    for category, v in zip(table[x], table["value"]):
        stems_x += [0, v, None]
        stems_y += [category, category, None]
    fig = go.Figure([
        go.Scatter(x=stems_x, y=stems_y, mode="lines", line={"color": "#9aa0a6", "width": 2}, hoverinfo="skip", showlegend=False),
        go.Scatter(x=table["value"], y=table[x], mode="markers", marker={"size": 12, "color": _PALETTE[0]}, name=label),
    ])
    fig.update_layout(title=f"{label} by {x}", xaxis_title=label, yaxis={"type": "category", "title": x}, showlegend=False)
    return fig


def _pie(df: pd.DataFrame, args: Any, hole: float = 0.0) -> go.Figure:
    kind = "donut" if hole else "pie"
    x = _need(args.x_column, kind, "x_column (the slice labels)")
    _exists(df, [x])
    value = args.value_column
    table = _aggregate(df, [x], value, args.agg_func if value else "count")
    if (table["value"] < 0).any():
        raise ValueError(f"{kind} slices cannot be negative; {value!r} has negative totals")
    table = table.sort_values("value", ascending=False)
    if len(table) > 15:
        rest = float(table["value"].iloc[14:].sum())
        table = pd.concat([table.iloc[:14], pd.DataFrame({x: ["Other"], "value": [rest]})], ignore_index=True)
    label = _value_label(args, value) if value else "count"
    fig = go.Figure(go.Pie(labels=table[x], values=table["value"], hole=hole, sort=False, textinfo="percent+label"))
    fig.update_layout(title=f"{label} by {x}")
    return fig


# ---------------------------------------------------------------------------
# Bivariate / multivariate
# ---------------------------------------------------------------------------


def _xy(df: pd.DataFrame, args: Any, kind: str) -> tuple[str, str]:
    x = _need(args.x_column, kind, "x_column")
    if not args.y_columns:
        raise ValueError(f"{kind} needs a numeric column in y_columns")
    y = args.y_columns[0]
    _numeric(df, [x, y])
    return x, y


def _joint(df: pd.DataFrame, args: Any) -> go.Figure:
    x, y = _xy(df, args, "joint")
    if args.color_column:
        _exists(df, [args.color_column])
    return px.scatter(
        df, x=x, y=y, color=args.color_column, marginal_x="histogram", marginal_y="histogram",
        opacity=0.7, title=f"Joint plot of {y} and {x}",
    )


def _hexbin(df: pd.DataFrame, args: Any) -> go.Figure:
    """Hexagonal binning on two offset lattices, the way matplotlib's hexbin does it."""
    x, y = _xy(df, args, "hexbin")
    pts = df[[x, y]].dropna().astype(float).to_numpy()
    if len(pts) < 2:
        raise ValueError("hexbin needs at least 2 rows with both values")
    xmin, ymin = pts.min(axis=0)
    xspan, yspan = np.ptp(pts, axis=0)
    xspan, yspan = xspan or 1.0, yspan or 1.0
    nx = args.bins
    ny = max(1, int(round(nx / math.sqrt(3))))
    u = (pts[:, 0] - xmin) / xspan * nx
    v = (pts[:, 1] - ymin) / yspan * ny
    i1, j1 = np.round(u), np.round(v)
    i2, j2 = np.floor(u), np.floor(v)
    d1 = (u - i1) ** 2 + 3 * (v - j1) ** 2
    d2 = (u - i2 - 0.5) ** 2 + 3 * (v - j2 - 0.5) ** 2
    first = d1 <= d2
    cu = np.where(first, i1, i2 + 0.5)
    cv = np.where(first, j1, j2 + 0.5)
    centres, counts = np.unique(np.column_stack([cu, cv]), axis=0, return_counts=True)
    corners_u = np.array([0.5, 0.0, -0.5, -0.5, 0.0, 0.5])
    corners_v = np.array([1 / 6, 1 / 3, 1 / 6, -1 / 6, -1 / 3, -1 / 6])
    levels = 8
    edges = np.linspace(counts.min(), counts.max() + 1e-9, levels + 1)
    level = np.clip(np.searchsorted(edges, counts, side="right") - 1, 0, levels - 1)
    colors = px.colors.sample_colorscale("Viridis", [k / (levels - 1) for k in range(levels)])
    fig = go.Figure()
    for k in range(levels):
        members = np.flatnonzero(level == k)
        if not len(members):
            continue
        xs: list[Any] = []
        ys: list[Any] = []
        for m in members:
            cx, cy = centres[m]
            xs += list(xmin + (cx + corners_u) / nx * xspan) + [xmin + (cx + corners_u[0]) / nx * xspan, None]
            ys += list(ymin + (cy + corners_v) / ny * yspan) + [ymin + (cy + corners_v[0]) / ny * yspan, None]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", fill="toself", fillcolor=colors[k],
            line={"color": colors[k], "width": 0.5}, hoverinfo="skip", showlegend=False,
        ))
    fig.add_trace(go.Scatter(
        x=xmin + centres[:, 0] / nx * xspan, y=ymin + centres[:, 1] / ny * yspan, mode="markers",
        marker={
            "size": 1, "opacity": 0, "color": counts, "colorscale": "Viridis",
            "cmin": float(counts.min()), "cmax": float(counts.max()),
            "colorbar": {"title": "count"}, "showscale": True,
        },
        text=counts, hovertemplate="count %{text}<extra></extra>", showlegend=False,
    ))
    fig.update_layout(title=f"Hexbin of {y} vs {x}", xaxis_title=x, yaxis_title=y)
    return fig


def _bubble(df: pd.DataFrame, args: Any) -> go.Figure:
    x, y = _xy(df, args, "bubble")
    size = _need(args.value_column, "bubble", "value_column (the bubble size)")
    _numeric(df, [size])
    if (df[size].dropna() < 0).any():
        raise ValueError(f"bubble sizes cannot be negative; {size!r} has negative values")
    if args.color_column:
        _exists(df, [args.color_column])
    frame = df.dropna(subset=[x, y, size])
    return px.scatter(
        frame, x=x, y=y, size=size, color=args.color_column, size_max=40, opacity=0.7,
        title=f"Bubble chart of {y} vs {x} (size: {size})",
    )


def _contour(df: pd.DataFrame, args: Any) -> go.Figure:
    x, y = _xy(df, args, "contour")
    if args.color_column:
        _exists(df, [args.color_column])
    return px.density_contour(
        df.dropna(subset=[x, y]), x=x, y=y, color=args.color_column,
        nbinsx=args.bins, nbinsy=args.bins, title=f"Density contours of {y} vs {x}",
    )


def _histogram_2d(df: pd.DataFrame, args: Any) -> go.Figure:
    x, y = _xy(df, args, "histogram_2d")
    value = args.value_column
    histfunc = {"count": "count", "sum": "sum", "mean": "avg", "min": "min", "max": "max"}
    if value:
        _numeric(df, [value])
        if args.agg_func not in histfunc:
            raise ValueError("histogram_2d supports agg_func count, sum, mean, min or max")
    frame = df.dropna(subset=[x, y])
    fig = px.density_heatmap(
        frame, x=x, y=y, z=value, histfunc=histfunc[args.agg_func] if value else "count",
        nbinsx=args.bins, nbinsy=args.bins, color_continuous_scale="Viridis",
        title=f"2D histogram of {y} vs {x}" + (f" ({args.agg_func} of {value})" if value else ""),
    )
    return fig


def _category_codes(series: pd.Series) -> tuple[np.ndarray, list[str]]:
    labels = _labels(series)
    names = sorted(labels.unique())
    lookup = {name: i for i, name in enumerate(names)}
    return labels.map(lookup).to_numpy(dtype=float), names


def _parallel(df: pd.DataFrame, args: Any) -> go.Figure:
    columns = _default_numeric(df, args, "parallel_coordinates", 2)
    frame = df[columns + ([args.color_column] if args.color_column else [])].dropna(subset=columns)
    line: dict[str, Any] = {}
    if args.color_column:
        _exists(df, [args.color_column])
        if _is_number(frame[args.color_column]):
            line = {"color": frame[args.color_column], "colorscale": "Viridis", "showscale": True,
                    "colorbar": {"title": args.color_column}}
        else:
            codes, names = _category_codes(frame[args.color_column])
            line = {"color": codes, "colorscale": "Turbo", "showscale": True,
                    "colorbar": {"title": args.color_column, "tickvals": list(range(len(names))), "ticktext": names}}
    fig = go.Figure(go.Parcoords(
        line=line,
        dimensions=[{"label": c, "values": frame[c].astype(float)} for c in columns],
    ))
    fig.update_layout(title="Parallel coordinates")
    return fig


def _minmax(frame: pd.DataFrame) -> pd.DataFrame:
    low, high = frame.min(), frame.max()
    return (frame - low) / (high - low).replace(0, 1)


def _radviz(df: pd.DataFrame, args: Any) -> go.Figure:
    columns = _default_numeric(df, args, "radviz", 3)
    frame = df[columns + ([args.color_column] if args.color_column else [])].dropna(subset=columns)
    scaled = _minmax(frame[columns].astype(float)).to_numpy()
    angles = 2 * np.pi * np.arange(len(columns)) / len(columns)
    anchors = np.column_stack([np.cos(angles), np.sin(angles)])
    weight = scaled.sum(axis=1, keepdims=True)
    points = np.divide(scaled @ anchors, weight, out=np.zeros((len(scaled), 2)), where=weight > 0)
    circle = np.linspace(0, 2 * np.pi, 120)
    fig = go.Figure(go.Scatter(
        x=np.cos(circle), y=np.sin(circle), mode="lines", line={"color": "#bbb"}, hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=anchors[:, 0], y=anchors[:, 1], mode="markers+text", text=columns, textposition="top center",
        marker={"size": 9, "color": "#444"}, hoverinfo="text", showlegend=False,
    ))
    groups = _groups(frame.reset_index(drop=True), args.color_column)
    for i, (label, rows) in enumerate(groups):
        fig.add_trace(go.Scatter(
            x=points[rows.index, 0], y=points[rows.index, 1], mode="markers", name=label or "rows",
            marker={"size": 6, "color": _PALETTE[i % len(_PALETTE)], "opacity": 0.75},
        ))
    fig.update_layout(
        title="RadViz", showlegend=bool(args.color_column),
        xaxis={"visible": False, "range": [-1.25, 1.25]},
        yaxis={"visible": False, "range": [-1.25, 1.25], "scaleanchor": "x"},
    )
    return fig


def _andrews(df: pd.DataFrame, args: Any) -> go.Figure:
    columns = _default_numeric(df, args, "andrews_curves", 2)
    frame = df[columns + ([args.color_column] if args.color_column else [])].dropna(subset=columns)
    if len(frame) > _MAX_CURVES:
        frame = frame.sample(n=_MAX_CURVES, random_state=42)
    frame = frame.reset_index(drop=True)
    scaled = _minmax(frame[columns].astype(float)).to_numpy()
    t = np.linspace(-np.pi, np.pi, 60)
    basis = [np.full_like(t, 1 / math.sqrt(2))]
    for k in range(1, len(columns)):
        harmonic = (k + 1) // 2
        basis.append(np.sin(harmonic * t) if k % 2 == 1 else np.cos(harmonic * t))
    curves = np.round(scaled @ np.array(basis), 4)
    t = np.round(t, 4)
    fig = go.Figure()
    for i, (label, rows) in enumerate(_groups(frame, args.color_column)):
        xs: list[Any] = []
        ys: list[Any] = []
        for r in rows.index:
            xs += list(t) + [None]
            ys += list(curves[r]) + [None]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name=label or "rows", opacity=0.5,
            line={"width": 1, "color": _PALETTE[i % len(_PALETTE)]},
        ))
    fig.update_layout(title="Andrews curves (min-max scaled)", xaxis_title="t", yaxis_title="f(t)",
                      showlegend=bool(args.color_column))
    return fig


# ---------------------------------------------------------------------------
# Time series / sequential
# ---------------------------------------------------------------------------


def _ordered_frame(df: pd.DataFrame, args: Any, kind: str) -> tuple[pd.DataFrame, str]:
    if not args.y_columns:
        raise ValueError(f"{kind} needs columns in y_columns")
    _numeric(df, args.y_columns)
    frame = df.copy()
    x = args.x_column
    if x:
        _exists(df, [x])
        if not (pd.api.types.is_numeric_dtype(frame[x]) or pd.api.types.is_datetime64_any_dtype(frame[x])):
            parsed = pd.to_datetime(frame[x], errors="coerce", format="mixed")
            if parsed.notna().sum() == frame[x].notna().sum():
                frame[x] = parsed
        if pd.api.types.is_numeric_dtype(frame[x]) or pd.api.types.is_datetime64_any_dtype(frame[x]):
            frame = frame.sort_values(x, kind="stable")
    else:
        x = "row"
        frame = frame.reset_index(drop=True)
        frame.insert(0, x, np.arange(len(frame)))
    return frame, x


def _line(df: pd.DataFrame, args: Any, area: bool = False) -> go.Figure:
    kind = "area" if area else "line"
    frame, x = _ordered_frame(df, args, kind)
    melted = frame[[x, *args.y_columns]].melt(id_vars=[x], var_name="series", value_name="value")
    draw = px.area if area else px.line
    return draw(melted, x=x, y="value", color="series", title=f"{'Area' if area else 'Line'} chart vs {x}")


def _candlestick(df: pd.DataFrame, args: Any) -> go.Figure:
    if len(args.y_columns) not in (1, 4):
        raise ValueError("candlestick needs 4 y_columns (open, high, low, close) or 1 column to summarise in windows")
    frame, x = _ordered_frame(df, args, "candlestick")
    if len(args.y_columns) == 4:
        o, h, low, c = args.y_columns
        fig = go.Figure(go.Candlestick(x=frame[x], open=frame[o], high=frame[h], low=frame[low], close=frame[c]))
        title = f"Candlestick ({o}, {h}, {low}, {c})"
    else:
        column = args.y_columns[0]
        values = frame[[x, column]].dropna()
        if len(values) < 2:
            raise ValueError("candlestick needs at least 2 values")
        windows = np.array_split(np.arange(len(values)), min(args.bins, len(values)))
        rows = [values.iloc[w] for w in windows if len(w)]
        fig = go.Figure(go.Candlestick(
            x=[r[x].iloc[0] for r in rows],
            open=[r[column].iloc[0] for r in rows], high=[r[column].max() for r in rows],
            low=[r[column].min() for r in rows], close=[r[column].iloc[-1] for r in rows],
        ))
        title = f"Candlestick of {column} in {len(rows)} windows (open, high, low, close)"
    fig.update_layout(title=title, xaxis_title=x, xaxis_rangeslider_visible=False)
    return fig


def _waterfall(df: pd.DataFrame, args: Any) -> go.Figure:
    x = _need(args.x_column, "waterfall", "x_column (the steps)")
    if not args.y_columns:
        raise ValueError("waterfall needs the numeric change column in y_columns")
    value = args.y_columns[0]
    _exists(df, [x])
    table = _aggregate(df, [x], value, "sum")
    if len(table) > _MAX_CATEGORIES:
        raise ValueError(f"waterfall shows at most {_MAX_CATEGORIES} steps; {x!r} has {len(table)}")
    fig = go.Figure(go.Waterfall(
        x=[*table[x], "Total"], y=[*table["value"], 0],
        measure=["relative"] * len(table) + ["total"],
        connector={"line": {"color": "#999"}},
    ))
    fig.update_layout(title=f"Waterfall of {value} by {x}", yaxis_title=value)
    fig.update_xaxes(type="category")
    return fig


def _correlogram(df: pd.DataFrame, args: Any, partial: bool) -> go.Figure:
    from statsmodels.tsa.stattools import acf, pacf

    kind = "pacf" if partial else "acf"
    column = _need(args.y_columns[0] if args.y_columns else args.x_column, kind, "a numeric column in y_columns")
    _numeric(df, [column])
    values = df[column].dropna().astype(float).to_numpy()
    if len(values) < 10:
        raise ValueError(f"{kind} needs at least 10 values; {column!r} has {len(values)}")
    lags = min(args.lags, len(values) // 2 - 1 if partial else len(values) - 1)
    if partial:
        result = pacf(values, nlags=lags, alpha=0.05, method="ywm", result_object=False)
    else:
        result = acf(values, nlags=lags, alpha=0.05, fft=True, result_object=False)
    coef, confint = np.asarray(result[0]), np.asarray(result[1])
    lag = np.arange(len(coef))
    upper = confint[:, 1] - coef
    lower = confint[:, 0] - coef
    stems_x: list[Any] = []
    stems_y: list[Any] = []
    for k, c in zip(lag, coef):
        stems_x += [k, k, None]
        stems_y += [0, c, None]
    fig = go.Figure([
        go.Scatter(x=lag[1:], y=upper[1:], mode="lines", line={"width": 0}, hoverinfo="skip", showlegend=False),
        go.Scatter(
            x=lag[1:], y=lower[1:], mode="lines", line={"width": 0}, fill="tonexty",
            fillcolor="rgba(99,110,250,0.18)", name="95% confidence band",
        ),
        go.Scatter(x=stems_x, y=stems_y, mode="lines", line={"color": "#636efa"}, hoverinfo="skip", showlegend=False),
        go.Scatter(x=lag, y=coef, mode="markers", marker={"color": "#636efa", "size": 7}, name=kind.upper()),
    ])
    name = "Partial autocorrelation" if partial else "Autocorrelation"
    fig.update_layout(title=f"{name} of {column}", xaxis_title="lag", yaxis_title=kind.upper(), yaxis_range=[-1.05, 1.05])
    return fig


# ---------------------------------------------------------------------------
# Hierarchical / relational
# ---------------------------------------------------------------------------


def _path(df: pd.DataFrame, args: Any, kind: str, minimum: int = 1) -> list[str]:
    columns = list(args.path_columns)
    if len(columns) < minimum:
        raise ValueError(f"{kind} needs at least {minimum} column(s) in path_columns")
    _exists(df, columns)
    return columns


def _hierarchy(df: pd.DataFrame, args: Any, sunburst: bool) -> go.Figure:
    kind = "sunburst" if sunburst else "treemap"
    path = _path(df, args, kind)
    frame = df[path].apply(_labels)
    value = args.value_column
    if value:
        _numeric(df, [value])
        weights = df[value].astype(float)
        if (weights.dropna() < 0).any():
            raise ValueError(f"{kind} sizes cannot be negative; {value!r} has negative values")
        frame["value"] = weights.fillna(0)
    else:
        frame["value"] = 1.0
    table = frame.groupby(path, sort=False)["value"].sum().reset_index()
    table = table[table["value"] > 0]
    draw = px.sunburst if sunburst else px.treemap
    return draw(
        table, path=path, values="value",
        title=f"{'Sunburst' if sunburst else 'Tree map'} of {' › '.join(path)}" + (f" (sum of {value})" if value else " (row counts)"),
    )


def _flows(df: pd.DataFrame, args: Any, path: list[str]) -> pd.DataFrame:
    value = args.value_column
    frames = []
    for a, b in zip(path, path[1:]):
        part = pd.DataFrame({"source": f"{a}: " + _labels(df[a]), "target": f"{b}: " + _labels(df[b])})
        if value:
            _numeric(df, [value])
            part["value"] = df[value].astype(float).fillna(0)
        else:
            part["value"] = 1.0
        frames.append(part.groupby(["source", "target"], sort=False)["value"].sum().reset_index())
    flows = pd.concat(frames, ignore_index=True)
    if value and (flows["value"] < 0).any():
        raise ValueError(f"Flows cannot be negative; {value!r} has negative totals")
    return flows[flows["value"] > 0]


def _sankey(df: pd.DataFrame, args: Any) -> go.Figure:
    path = _path(df, args, "sankey", 2)
    flows = _flows(df, args, path)
    nodes = list(dict.fromkeys([*flows["source"], *flows["target"]]))
    if len(nodes) > 2 * _MAX_NODES:
        raise ValueError(f"sankey would have {len(nodes)} nodes; choose columns with fewer distinct values")
    index = {n: i for i, n in enumerate(nodes)}
    fig = go.Figure(go.Sankey(
        node={"label": nodes, "pad": 12, "thickness": 16},
        link={"source": flows["source"].map(index), "target": flows["target"].map(index), "value": flows["value"]},
    ))
    fig.update_layout(title=f"Sankey diagram: {' → '.join(path)}" + (f" (sum of {args.value_column})" if args.value_column else ""))
    return fig


def _edges(df: pd.DataFrame, args: Any, kind: str) -> tuple[pd.DataFrame, str]:
    """Weighted edges from two path_columns, or a correlation network of numeric columns."""
    if args.path_columns:
        if len(args.path_columns) != 2:
            raise ValueError(f"{kind} needs exactly 2 path_columns (source, target), or none for a correlation network")
        source, target = _path(df, args, kind, 2)
        value = args.value_column
        frame = pd.DataFrame({"source": _labels(df[source]), "target": _labels(df[target])})
        if value:
            _numeric(df, [value])
            frame["weight"] = df[value].astype(float).fillna(0)
        else:
            frame["weight"] = 1.0
        edges = frame.groupby(["source", "target"], sort=False)["weight"].sum().reset_index()
        edges = edges[edges["weight"] != 0]
        return edges, f"{source} → {target}" + (f" (sum of {value})" if value else " (row counts)")
    columns = _default_numeric(df, args, kind, 2, limit=30)
    corr = df[columns].astype(float).corr()
    rows = [
        {"source": a, "target": b, "weight": float(corr.loc[a, b])}
        for i, a in enumerate(columns)
        for b in columns[i + 1:]
        if np.isfinite(corr.loc[a, b]) and abs(corr.loc[a, b]) >= args.threshold
    ]
    if not rows:
        raise ValueError(f"No column pairs have |correlation| ≥ {args.threshold}; lower threshold")
    return pd.DataFrame(rows), f"correlations with |r| ≥ {args.threshold}"


def _spring_layout(nodes: list[str], edges: pd.DataFrame, seed: int = 42, iterations: int = 200) -> np.ndarray:
    """Fruchterman-Reingold force-directed positions in the unit square."""
    n = len(nodes)
    rng = np.random.default_rng(seed)
    pos = rng.random((n, 2))
    if n == 1:
        return pos
    index = {name: i for i, name in enumerate(nodes)}
    a = edges["source"].map(index).to_numpy()
    b = edges["target"].map(index).to_numpy()
    w = np.abs(edges["weight"].to_numpy(dtype=float))
    w = w / w.max() if w.max() > 0 else np.ones_like(w)
    k = 1 / math.sqrt(n)
    temperature = 0.1
    for _ in range(iterations):
        delta = pos[:, None, :] - pos[None, :, :]
        distance = np.maximum(np.linalg.norm(delta, axis=2), 1e-4)
        force = (delta / distance[..., None] * (k * k / distance)[..., None]).sum(axis=1)
        pull = pos[a] - pos[b]
        length = np.maximum(np.linalg.norm(pull, axis=1), 1e-4)
        attract = pull * (length * w / k)[:, None]
        np.add.at(force, a, -attract)
        np.add.at(force, b, attract)
        size = np.maximum(np.linalg.norm(force, axis=1), 1e-9)
        pos += force / size[:, None] * np.minimum(size, temperature)[:, None]
        temperature *= 0.98
    pos -= pos.min(axis=0)
    span = pos.max(axis=0)
    return pos / np.where(span > 0, span, 1)


def _network(df: pd.DataFrame, args: Any) -> go.Figure:
    edges, described = _edges(df, args, "network")
    nodes = list(dict.fromkeys([*edges["source"], *edges["target"]]))
    if len(nodes) > _MAX_NODES:
        raise ValueError(f"network would have {len(nodes)} nodes (at most {_MAX_NODES}); choose columns with fewer distinct values")
    pos = _spring_layout(nodes, edges)
    index = {name: i for i, name in enumerate(nodes)}
    signed = not args.path_columns
    fig = go.Figure()
    for positive in ((True, False) if signed else (True,)):
        part = edges[(edges["weight"] > 0) == positive] if signed else edges
        xs: list[Any] = []
        ys: list[Any] = []
        for s, t in zip(part["source"], part["target"]):
            xs += [pos[index[s], 0], pos[index[t], 0], None]
            ys += [pos[index[s], 1], pos[index[t], 1], None]
        if xs:
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", hoverinfo="skip",
                name=("positive" if positive else "negative") if signed else "edges",
                line={"width": 1.5, "color": ("#ef553b" if positive else "#636efa") if signed else "#9aa0a6"},
            ))
    degree = pd.concat([edges["source"], edges["target"]]).value_counts()
    fig.add_trace(go.Scatter(
        x=pos[:, 0], y=pos[:, 1], mode="markers+text", text=nodes, textposition="top center",
        marker={"size": [8 + 3 * min(int(degree.get(n, 0)), 10) for n in nodes], "color": "#00cc96",
                "line": {"width": 1, "color": "#333"}},
        hovertext=[f"{n}: {int(degree.get(n, 0))} connection(s)" for n in nodes], hoverinfo="text", name="nodes",
    ))
    fig.update_layout(
        title=f"Network graph — {described}", showlegend=signed,
        xaxis={"visible": False}, yaxis={"visible": False},
    )
    return fig


def _dendrogram(df: pd.DataFrame, args: Any) -> go.Figure:
    import plotly.figure_factory as ff
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import pdist

    columns = _default_numeric(df, args, "dendrogram", 2, limit=30)
    frame = df[columns].dropna().astype(float)
    constant = [c for c in columns if frame[c].nunique() <= 1]
    columns = [c for c in columns if c not in constant]
    if len(columns) < 2 or len(frame) < 3:
        raise ValueError("dendrogram needs at least 2 non-constant numeric columns and 3 complete rows")
    data = frame[columns].to_numpy().T
    fig = ff.create_dendrogram(
        data, labels=columns,
        distfun=lambda m: np.clip(pdist(m, "correlation"), 0, None),
        linkagefun=lambda d: linkage(d, "average"),
    )
    fig.update_layout(
        title="Dendrogram of the columns (average linkage, distance 1 − correlation)",
        yaxis_title="distance", width=None, height=None, autosize=True,
    )
    return fig


def _chord(df: pd.DataFrame, args: Any) -> go.Figure:
    edges, described = _edges(df, args, "chord")
    nodes = list(dict.fromkeys([*edges["source"], *edges["target"]]))
    if len(nodes) > 60:
        raise ValueError(f"chord would have {len(nodes)} nodes (at most 60); choose columns with fewer distinct values")
    weight = edges["weight"].abs()
    totals = pd.concat([
        pd.Series(weight.to_numpy(), index=edges["source"]),
        pd.Series(weight.to_numpy(), index=edges["target"]),
    ]).groupby(level=0).sum().reindex(nodes).fillna(0)
    gap = 0.02 * 2 * np.pi
    usable = 2 * np.pi - gap * len(nodes)
    spans = totals / totals.sum() * usable
    starts = np.concatenate([[0], np.cumsum(spans.to_numpy() + gap)[:-1]])
    mid = {n: starts[i] + spans.iloc[i] / 2 for i, n in enumerate(nodes)}
    colors = {n: _PALETTE[i % len(_PALETTE)] for i, n in enumerate(nodes)}
    fig = go.Figure()
    for i, n in enumerate(nodes):
        theta = np.linspace(starts[i], starts[i] + spans.iloc[i], 30)
        fig.add_trace(go.Scatter(
            x=np.cos(theta), y=np.sin(theta), mode="lines", line={"width": 12, "color": colors[n]},
            name=n, hovertext=f"{n}: total {totals[n]:.4g}", hoverinfo="text",
        ))
        fig.add_annotation(x=1.14 * math.cos(mid[n]), y=1.14 * math.sin(mid[n]), text=n, showarrow=False, font={"size": 10})
    top = float(weight.max()) or 1.0
    t = np.linspace(0, 1, 30)
    for (s, tg, w) in zip(edges["source"], edges["target"], edges["weight"]):
        p0 = np.array([math.cos(mid[s]), math.sin(mid[s])]) * 0.97
        p2 = np.array([math.cos(mid[tg]), math.sin(mid[tg])]) * 0.97
        curve = ((1 - t) ** 2)[:, None] * p0 + (t ** 2)[:, None] * p2
        fig.add_trace(go.Scatter(
            x=curve[:, 0], y=curve[:, 1], mode="lines", opacity=0.55, showlegend=False,
            line={"width": 1 + 7 * abs(w) / top, "color": colors[s], "dash": "dot" if w < 0 else "solid"},
            hovertext=f"{s} — {tg}: {w:.4g}", hoverinfo="text",
        ))
    fig.update_layout(
        title=f"Chord diagram — {described}", showlegend=False,
        xaxis={"visible": False, "range": [-1.35, 1.35]},
        yaxis={"visible": False, "range": [-1.35, 1.35], "scaleanchor": "x"},
    )
    return fig


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def _silhouette(df: pd.DataFrame, args: Any) -> go.Figure:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples
    from sklearn.preprocessing import StandardScaler

    columns = _default_numeric(df, args, "silhouette", 1, limit=10)
    frame = df[columns].dropna().astype(float)
    if len(frame) <= args.n_clusters:
        raise ValueError(f"silhouette needs more complete rows ({len(frame)}) than n_clusters ({args.n_clusters})")
    X = StandardScaler().fit_transform(frame.to_numpy())
    labels = KMeans(n_clusters=args.n_clusters, n_init=10, random_state=42).fit_predict(X)
    if len(set(labels)) < 2:
        raise ValueError("KMeans found fewer than 2 clusters; the rows may be identical")
    scores = silhouette_samples(X, labels)
    average = float(scores.mean())
    fig = go.Figure()
    bottom = 0
    for cluster in range(args.n_clusters):
        values = np.sort(scores[labels == cluster])
        if not len(values):
            continue
        y = np.arange(bottom, bottom + len(values))
        fig.add_trace(go.Scatter(
            x=np.concatenate([[0], values, [0]]), y=np.concatenate([[y[0]], y, [y[-1]]]),
            mode="lines", fill="toself", name=f"cluster {cluster} ({len(values)})",
            line={"width": 0.5, "color": _PALETTE[cluster % len(_PALETTE)]},
        ))
        bottom += len(values) + max(2, len(frame) // 50)
    fig.add_vline(x=average, line_dash="dash", line_color="#d62728", annotation_text=f"mean {average:.3f}")
    fig.update_layout(
        title=f"Silhouette plot — KMeans with {args.n_clusters} clusters on {len(columns)} standardised column(s)",
        xaxis_title="silhouette coefficient", yaxis={"visible": False}, xaxis_range=[-1, 1],
    )
    return fig


_BUILDERS: dict[str, Callable[[pd.DataFrame, Any], go.Figure]] = {
    "kde": _kde,
    "rug": _rug,
    "ridgeline": _ridgeline,
    "ecdf": _ecdf,
    "qq_plot": _qq,
    "bar": _bar,
    "count": _count,
    "swarm": _swarm,
    "strip": _strip,
    "point": _point,
    "lollipop": _lollipop,
    "pie": _pie,
    "donut": lambda df, args: _pie(df, args, hole=0.45),
    "joint": _joint,
    "hexbin": _hexbin,
    "bubble": _bubble,
    "contour": _contour,
    "histogram_2d": _histogram_2d,
    "parallel_coordinates": _parallel,
    "radviz": _radviz,
    "andrews_curves": _andrews,
    "line": _line,
    "area": lambda df, args: _line(df, args, area=True),
    "candlestick": _candlestick,
    "waterfall": _waterfall,
    "acf": lambda df, args: _correlogram(df, args, partial=False),
    "pacf": lambda df, args: _correlogram(df, args, partial=True),
    "treemap": lambda df, args: _hierarchy(df, args, sunburst=False),
    "sunburst": lambda df, args: _hierarchy(df, args, sunburst=True),
    "sankey": _sankey,
    "network": _network,
    "dendrogram": _dendrogram,
    "chord": _chord,
    "silhouette": _silhouette,
}

assert set(_BUILDERS) == set(EXTRA_PLOT_TYPES)


def build_extra_figure(df: pd.DataFrame, args: Any) -> go.Figure:
    return _BUILDERS[args.plot_type.value](df, args)
