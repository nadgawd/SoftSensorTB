"""Regression tests for intent routing.

The rule layer short-circuits before any LLM call, so a wrong regex is a wrong
intent no matter how good the router model is. These cases pin that layer down.
"""

from __future__ import annotations

import asyncio

import pytest

from backend.agents.router import _rule_based_intent, classify_intent

# Turns the regex layer must settle on its own, without consulting a model.
RULE_CASES = [
    # --- theory: must not be dragged into EXECUTE by an incidental keyword ---
    ("what is a violin plot", "RAG"),
    ("what is a soft sensor", "RAG"),
    ("what is PLS", "RAG"),
    ("what is r2", "RAG"),
    ("what is multicollinearity", "RAG"),
    ("explain PCA", "RAG"),
    ("how does PLS handle collinearity", "RAG"),
    ("why should one standardise inputs", "RAG"),
    ("define overfitting", "RAG"),
    ("difference between PLS and PCR", "RAG"),
    ("compare OLS and PLS", "RAG"),
    ("when should i use ridge", "RAG"),
    ("pros and cons of lasso", "RAG"),
    ("recommend an algorithm", "RAG"),
    ("which model should i use", "RAG"),
    # --- theory phrasings that previously leaked into EXECUTE ---
    ("describe PLS regression", "RAG"),          # "describe" is also an action verb
    ("summarize how PLS works", "RAG"),          # "summar" is also an action verb
    ("on average how well does PLS perform", "RAG"),  # "average" is also an action verb
    ("tell me about soft sensors", "RAG"),
    ("so what is a soft sensor", "RAG"),         # lead-in must not skip the ^ anchor
    ("in general how does PCA work", "RAG"),
    # --- data commands ---
    ("plot DT vs S", "EXECUTE"),
    ("show a violin plot of t", "EXECUTE"),
    ("histogram of feed_rate", "EXECUTE"),
    ("drop rows where T > 300", "EXECUTE"),
    ("train a PLS model for quality", "EXECUTE"),
    ("remove outliers from T_C", "EXECUTE"),
    ("what is the max value of u1", "EXECUTE"),
    ("how many nulls are there", "EXECUTE"),
    ("why is there a spike in the chart", "EXECUTE"),
    ("normalize my columns", "EXECUTE"),
    # --- data phrasings that reuse theory verbs; the lookaheads must hold ---
    ("explain this plot", "EXECUTE"),
    ("explain my residuals", "EXECUTE"),
    ("describe my dataset", "EXECUTE"),
    ("describe the distribution of T_C", "EXECUTE"),
    ("describe u1", "EXECUTE"),                  # a column, not a concept
    ("summarize the model coefficients", "EXECUTE"),
    ("compare my two models", "EXECUTE"),
    ("how do i drop nulls", "EXECUTE"),
    ("how well does my model perform", "EXECUTE"),
    ("just drop the duplicate rows", "EXECUTE"),  # lead-in must not swallow the verb
]

# Genuinely ambiguous turns belong to the model, not the regexes.
AMBIGUOUS = [
    "is PLS better than PCR for my data",
    "tell me about my data",
]


@pytest.mark.parametrize("utterance,expected", RULE_CASES)
def test_rule_layer_classifies(utterance: str, expected: str) -> None:
    assert _rule_based_intent(utterance) == expected


@pytest.mark.parametrize("utterance", AMBIGUOUS)
def test_ambiguous_turns_defer_to_model(utterance: str) -> None:
    assert _rule_based_intent(utterance) is None


@pytest.mark.parametrize("utterance,expected", RULE_CASES)
def test_rule_hits_never_reach_the_model(utterance: str, expected: str, monkeypatch) -> None:
    """A rule-decided turn must not cost an LLM round-trip."""

    def _boom(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("router consulted the LLM for a rule-decided turn")

    monkeypatch.setattr("backend.agents.router.get_llm_client", _boom)
    assert asyncio.run(classify_intent(utterance, has_dataset=True)) == expected


@pytest.mark.parametrize(
    "has_dataset,expected",
    [(True, "EXECUTE"), (False, "RAG")],
)
def test_falls_back_when_model_unavailable(has_dataset: bool, expected: str, monkeypatch) -> None:
    """A dead router model must not break chat; it falls back on dataset presence."""

    def _unavailable(*args, **kwargs):
        raise RuntimeError("no providers configured")

    monkeypatch.setattr("backend.agents.router.get_llm_client", _unavailable)
    intent = asyncio.run(classify_intent(AMBIGUOUS[0], has_dataset=has_dataset))
    assert intent == expected


def test_only_valid_intents_are_returned(monkeypatch) -> None:
    """Garbage from the router model is coerced to the fallback, never passed through."""

    class _Junk:
        class choices_0:
            class message:
                content = "banana"

        choices = [choices_0]

    class _Completions:
        async def create(self, **kwargs):
            return _Junk()

    class _Client:
        chat = type("chat", (), {"completions": _Completions()})()

    monkeypatch.setattr("backend.agents.router.get_llm_client", lambda **kw: _Client())
    intent = asyncio.run(classify_intent(AMBIGUOUS[0], has_dataset=True))
    assert intent in {"EXECUTE", "RAG"}
