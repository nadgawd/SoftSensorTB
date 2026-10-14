"""Which tool schemas the execution agent is sent each round.

All the schemas together are ~80k characters, too much to resend every round
to small or rate-limited models. Each turn gets the core tools, the tools whose
keywords appear in the request (or, when none do, the tools of the user's
current pipeline step), and the tools already used in this conversation.
Every other tool is named in
the system prompt, and calling one adds its schema from the next round on, so
no tool is out of reach.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional

CORE_TOOLS = (
    "inspect_data", "get_column_statistics", "run_duckdb_query", "generate_custom_plot",
    "list_trained_models", "set_model_features",
)

_STEP_LABELS = {
    "data ingestion": "ingestion",
    "ingestion": "ingestion",
    "data quality": "quality",
    "quality": "quality",
    "preprocessing": "preprocessing",
    "feature selection": "feature-selection",
    "feature-selection": "feature-selection",
    "modeling": "modeling",
    "modelling": "modeling",
    "validation": "validation",
    "export": "export",
}

# (pattern, tools): a request matching the pattern gets these tools.
_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (r"plot|chart|graph|visuali[sz]|histogram|heatmap|scatter|draw|diagram|\bmap\b|curve|kde|density|violin|"
     r"box ?plot|treemap|sunburst|sankey|dendrogram|ecdf|q-?q|acf|pacf|autocorrelation", ("generate_custom_plot",)),
    (r"group ?by|pivot|crosstab|cross[- ]tab|aggregat|per (group|category)|by category", ("aggregate_data",)),
    (r"t[- ]?test|anova|chi[- ]?sq|χ|hypothes|p[- ]?value|significan|normality|shapiro|kolmogorov|\bks\b|"
     r"stationar|dickey|\badf\b|unit root", ("statistical_test",)),
    (r"\bsort|order by|ascending|descending|rank rows", ("sort_data",)),
    (r"\bmelt|\bstack|unstack|reshape|long format|wide format|to long|to wide|\bpivot\b", ("reshape_data",)),
    (r"merge|\bjoin|concat|combine|append|another (file|dataset)|second (file|dataset)|other (file|dataset)",
     ("combine_datasets", "list_datasets")),
    (r"datasets?\b.*(uploaded|have)|uploaded files?|list (my )?(datasets|files)", ("list_datasets",)),
    (r"\bmask\b|\bwhere\(|replace values where|set .* where", ("mask_values",)),
    (r"\bmap\b|recode|relabel|map values|replace .* with", ("map_values", "clean_text")),
    (r"\bapply\b|transform\(|cumsum|cumulative|\brank|z-?score|demean|group mean|\babs\b|\bround\b|floor|ceil",
     ("apply_function",)),
    (r"tf-?idf|vectori[sz]|count ?vectori|hashing|bag of words|word2vec|glove|embedding|\blsa\b|text features",
     ("vectorize_text",)),
    (r"haversine|distance|latitude|longitude|\blat\b|\blon\b|geo|coordinates", ("geo_features",)),
    (r"shap|permutation|tree[- ]based|feature importance|importances?\b", ("select_features_model", "generate_importance_plot")),
    (r"missing|impute|imputation|\bfill|interpolat|\bnan\b|\bnull|mice|knn imput", ("remove_missing_data",)),
    (r"outlier|winsor|\bclip|\bcap\b", ("remove_outliers",)),
    (r"anomal|isolation forest", ("detect_anomalies",)),
    (r"duplicate", ("drop_duplicates",)),
    (r"filter|query|drop rows|remove rows|keep rows|rows where|boolean index", ("drop_rows",)),
    (r"convert|dtype|parse dates?|to datetime|to numeric|as type|astype", ("convert_types",)),
    (r"strip|lower ?case|upper ?case|whitespace|regex|str\.|contains|extract|text clean", ("clean_text",)),
    (r"drop col|remove col|delete col", ("drop_columns",)),
    (r"rename", ("rename_columns",)),
    (r"normali[sz]|standardi[sz]|scal(e|ing)|min-?max|robust scal", ("normalize_data",)),
    (r"\blog\b|box-?cox|yeo|sqrt|square root|reciprocal|\bbin|binari[sz]|discreti[sz]|squared?\b",
     ("transform_features",)),
    (r"encod|one-?hot|dummies|label encod|ordinal|woe|weight of evidence|leave-?one-?out enc|target enc",
     ("encode_categorical",)),
    (r"rolling|moving average|window|expanding|\bewm|exponential", ("rolling_aggregate",)),
    (r"\blag|\blead|shift|\bdiff|differenc|pct[_ ]change|percent change", ("create_lag_features",)),
    (r"polynomial|interaction|day of week|weekend|holiday|quarter|datetime feature|date feature|cyclical|"
     r"ratio|pairwise|feature engineer|product of", ("feature_engineer",)),
    (r"formula|new column|compute column|derive", ("execute_formula",)),
    (r"smote|adasyn|oversampl|undersampl|imbalanc|balanc", ("balance_data",)),
    (r"\brfe\b|recursive feature", ("select_features_rfe",)),
    (r"k ?best|selectkbest|mutual info|f[_ ]regression", ("select_k_best",)),
    (r"collinear|multicollinear|\bvif\b", ("drop_collinear_features",)),
    (r"\bpca\b|principal component|svd|\blda\b|discriminant|t-?sne|umap|autoencoder|dimension", ("reduce_dimensions",)),
    (r"train|fit|model|regress|\bpls\b|ridge|lasso|\bknn\b|k-nn|\bols\b|pcr|cross[- ]?valid|\bcv\b|k-?fold|"
     r"split|holdout|validation set|bootstrap|out-of-bag|\boob\b|walk[- ]forward|loocv|leave[- ]one", ("train_soft_sensor",)),
    (r"parity|predicted vs|actual vs", ("generate_parity_plot",)),
    (r"residual", ("generate_residuals_plot",)),
    (r"coefficient", ("generate_coefficients_plot",)),
    (r"\broc\b|auc|receiver operating", ("generate_roc_curve",)),
    (r"precision|recall|\bpr curve|average precision", ("generate_pr_curve",)),
    (r"learning curve|more data|over-?fit|under-?fit|bias.variance", ("generate_learning_curve",)),
)
_COMPILED = tuple((re.compile(pattern, re.IGNORECASE), tools) for pattern, tools in _KEYWORDS)


def step_id(step_hint: Optional[str]) -> Optional[str]:
    if not step_hint:
        return None
    return _STEP_LABELS.get(step_hint.strip().lower())


def _step_tools(step: Optional[str]) -> list[str]:
    if step is None:
        return []
    # Imported here: backend.api.tools imports the execution agent, which imports this module.
    from backend.api.tools import tools_for_step

    return tools_for_step(step)


def keyword_tools(text: str) -> list[str]:
    out: list[str] = []
    for pattern, tools in _COMPILED:
        if pattern.search(text or ""):
            out.extend(tools)
    return out


def select_tool_names(
    available: Iterable[str],
    *,
    user_input: str,
    step_hint: Optional[str] = None,
    used: Iterable[str] = (),
) -> list[str]:
    """The tool names to send, in ``available`` order.

    A request that names an operation gets the core tools plus the matching
    ones; a vague one ("what next?") gets the tools of the current step.
    """
    names = list(available)
    step = step_id(step_hint)
    if step is None:
        return names
    matched = keyword_tools(user_input)
    wanted = {*CORE_TOOLS, *(matched or _step_tools(step)), *used}
    return [n for n in names if n in wanted]


def select_schemas(
    schemas: list[dict[str, Any]],
    *,
    user_input: str,
    step_hint: Optional[str] = None,
    used: Iterable[str] = (),
) -> tuple[list[dict[str, Any]], list[str]]:
    """(schemas to send, names of the tools left out)."""
    names = [s["function"]["name"] for s in schemas]
    chosen = set(select_tool_names(names, user_input=user_input, step_hint=step_hint, used=used))
    return [s for s in schemas if s["function"]["name"] in chosen], [n for n in names if n not in chosen]


__all__ = ["CORE_TOOLS", "keyword_tools", "select_schemas", "select_tool_names", "step_id"]
