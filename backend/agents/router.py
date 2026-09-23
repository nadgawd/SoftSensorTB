"""Intent router agent — routes user turns to EXECUTE vs RAG."""

from __future__ import annotations

import logging
import re

from backend.agents.llm_client import get_llm_client

logger = logging.getLogger(__name__)

ROUTER_SYSTEM_PROMPT = (
    "You classify user messages into exactly one of: EXECUTE or RAG.\n"
    "Output EXECUTE if the user wants to run SQL, modify data, plot/draw charts, "
    "train models, OR IF THEY ASK QUESTIONS ABOUT THEIR DATA/GRAPHS (e.g. 'what is the max temperature?', "
    "'how many nulls?', 'why is there a spike in the chart?').\n"
    "Output RAG if the user asks purely conceptual or general engineering questions "
    "(e.g. 'what is a soft sensor', 'how does PCA work', 'explain PLS', 'recommend an algorithm'), "
    "or any theoretical request that does NOT require querying their specific dataset.\n"
    "Short commands naming columns are always EXECUTE, even without a verb.\n"
    "Examples:\n"
    "  'plot DT vs S' -> EXECUTE\n"
    "  'show a violin plot of t' -> EXECUTE\n"
    "  'histogram of feed_rate' -> EXECUTE\n"
    "  'drop rows where T > 300' -> EXECUTE\n"
    "  'what is a violin plot' -> RAG\n"
    "  'how does PLS handle collinearity' -> RAG\n"
    "When in doubt about the user's specific data, output EXECUTE. Output only the single word EXECUTE or RAG."
)

_VALID_INTENTS = frozenset({"EXECUTE", "RAG"})

# Domain vocabulary that only ever shows up in theory questions. Gating the
# "describe"/"summarise" cues on these keeps "describe PLS" (theory) apart from
# "describe u1", where the object is a column in the user's own dataset.
_CONCEPT_TERMS = (
    r"pls|pca|pcr|ols|mlr|knn|ridge|lasso|rfe|smote|vif|r2|rmse|mae|"
    r"soft[\s-]?sensors?|regression|collinearity|multi\w*|"
    r"overfitting|underfitting|normali[sz]ation|standardi[sz]ation|"
    r"cross[\s-]?validation|dimensionality\s+reduction|feature\s+engineering"
)

# Discourse lead-ins, skipped so they cannot push a theory question past the
# ^ anchor — without this, "on average, how well does PLS perform?" reaches
# _ACTION_RE and matches on "average".
_LEAD_IN = (
    r"(?:(?:so|ok|okay|well|hey|also|and|but|now|just|please|"
    r"quick\s+question|on\s+average|in\s+general|generally|typically|usually)"
    r"\b[\s,.:;-]*)*"
)

# Referents that make a request data-grounded rather than conceptual.
_MINE = r"(?:this|that|the|my|these|those|it)\b"

# Conceptual cues are checked first: "what is a violin plot" is theory even though
# it names a chart type. The lookaheads keep data-grounded phrasings ("explain this
# plot", "how do I drop nulls") out of the RAG bucket.
_CONCEPT_RE = re.compile(
    rf"^\s*{_LEAD_IN}(?:"
    r"what\s+(?:is|are)\s+(?:a|an)\b"
    rf"|what\s+(?:is|are)\s+(?:{_CONCEPT_TERMS})\b"
    r"|(?:what|which)\s+(?:algorithm|model|method|technique)s?\b"
    rf"|explain\s+(?!{_MINE})"
    rf"|(?:describe|summari[sz]e|tell\s+me\s+about)\s+(?:how\s+)?(?:an?\s+)?"
    rf"(?:{_CONCEPT_TERMS})\b"
    rf"|how\s+well\s+(?:does|do)\s+(?!{_MINE})"
    r"|how\s+(?:does|do|is|are)\s+(?!(?:i|we|my|this|that|the)\b)"
    r"|why\s+(?:is|are|would|should)\s+(?:one|someone|we|you)\b"
    r"|define\b"
    r"|difference\s+between\b"
    r"|compare\s+(?!(?:this|the|my)\b)"
    r"|when\s+should\s+(?:i|we|one)\b"
    r"|pros\s+and\s+cons\b"
    r"|recommend\b|suggest\s+(?:an?|some)\b"
    r")",
    re.IGNORECASE,
)

# Verbs and nouns that only make sense against the loaded dataset.
_ACTION_RE = re.compile(
    r"\b(?:"
    r"plot\w*|chart\w*|graph\w*|visuali[sz]\w*|draw|render|display|"
    r"histogram\w*|scatter\w*|violin\w*|box[\s-]?plot\w*|heat[\s-]?map\w*|"
    r"pair[\s-]?plot\w*|line[\s-]?plot\w*|bar[\s-]?chart\w*|parity|residual\w*|"
    r"train\w*|fit|refit|model\w*|predict\w*|validat\w*|cross[\s-]?val\w*|"
    r"remove|drop\w*|delete|impute\w*|fillna|"
    r"normali[sz]\w*|standardi[sz]\w*|scal\w*|encode\w*|transform\w*|"
    r"engineer\w*|balance\w*|rename|resample\w*|rolling|lag\w*|smooth\w*|"
    r"anomal\w*|outlier\w*|"
    r"quer\w*|sql|duckdb|select|unselect\w*|deselect\w*|feature\w*|"
    r"count|sum|average|mean|median|max|min|std|"
    r"correlat\w*|describe|summar\w*|statistic\w*|null\w*|missing|dupli\w*|"
    r"column\w*|row\w*|dataset|undo|revert|snapshot"
    r")\b",
    re.IGNORECASE,
)


def _rule_based_intent(user_input: str) -> str | None:
    """Return a confident intent for unambiguous turns, else ``None``."""
    if _CONCEPT_RE.search(user_input):
        return "RAG"
    if _ACTION_RE.search(user_input):
        return "EXECUTE"
    return None


async def classify_intent(
    user_input: str,
    has_dataset: bool = False,
    previous_intent: str | None = None,
) -> str:
    """
    Classify ``user_input`` as ``EXECUTE`` or ``RAG``.

    Unambiguous turns are decided by keyword rules so plotting and data commands
    never depend on a live LLM call. A turn with no cue either way that follows
    an EXECUTE turn is a follow-up ("fetch the list", "yes, do it") and stays
    EXECUTE. Everything else goes to the router model, which falls back to
    ``EXECUTE`` when a dataset is loaded.
    """
    rule_intent = _rule_based_intent(user_input)
    if rule_intent is not None:
        return rule_intent
    if has_dataset and previous_intent == "EXECUTE":
        return "EXECUTE"

    fallback = "EXECUTE" if has_dataset else "RAG"

    try:
        client = get_llm_client(agent_type="router")
        response = await client.chat.completions.create(
            temperature=0,
            max_tokens=16,
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ],
        )
        raw = (response.choices[0].message.content or "").strip().upper()
        match = re.search(r"\b(EXECUTE|RAG)\b", raw)
        if match and match.group(1) in _VALID_INTENTS:
            return match.group(1)
        logger.warning("Router returned unparseable intent %r; using %s", raw, fallback)
    except Exception:
        logger.exception("Router LLM call failed; using %s", fallback)

    return fallback
