"""Whitelist check for the pandas expressions users and the model may send.

``DataFrame.eval`` and ``DataFrame.query`` resolve ``@name`` against the
caller's local variables and evaluate attribute access and method calls on
columns (``x.to_csv('/path')``), so an expression must be limited to column
names, literals, arithmetic, comparisons, boolean logic and pandas' math
functions before it is evaluated.
"""

from __future__ import annotations

import ast
import re

from pandas.core.computation.ops import MATHOPS

MAX_EXPRESSION_CHARS = 1000

FUNCTIONS = tuple(MATHOPS)
_FUNCTIONS = frozenset(MATHOPS)
_BACKTICKED = re.compile(r"`[^`]*`")
_BRACKETED = re.compile(r"\[([^\[\]`'\"]+)\]")
_LIST_CONTEXT = re.compile(r"(?:\bin|==|!=)\s*$")
_QUOTED = re.compile(r"(`[^`]*`|\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*')")


def bracketed_columns(text: str) -> str:
    """``[Feed temp]`` → ``\\`Feed temp\\```: square brackets around one name mean a column.

    A bracket after ``in``, ``==`` or ``!=``, or holding literals or several items, stays a list.
    """
    def convert(part: str) -> str:
        def replace(match: re.Match[str]) -> str:
            inner = match.group(1).strip()
            if not inner or _LIST_CONTEXT.search(part[: match.start()]):
                return match.group(0)
            try:
                items = ast.parse(f"[{inner}]", mode="eval").body.elts  # type: ignore[attr-defined]
            except SyntaxError:
                return f"`{inner}`"
            if len(items) == 1 and isinstance(items[0], ast.Name):
                return f"`{inner}`"
            return match.group(0)

        return _BRACKETED.sub(replace, part)

    parts = _QUOTED.split(text)
    return "".join(part if i % 2 else convert(part) for i, part in enumerate(parts))
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Name, ast.Load,
    ast.Constant, ast.List, ast.Tuple, ast.Call,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.BitAnd, ast.BitOr, ast.BitXor, ast.Invert, ast.Not, ast.UAdd, ast.USub,
    ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
)


def assert_safe_expression(expression: str, *, kind: str = "expression") -> str:
    """Return the expression stripped, or raise ValueError naming what is not allowed."""
    text = bracketed_columns((expression or "").strip())
    if not text:
        raise ValueError(f"The {kind} is empty")
    if len(text) > MAX_EXPRESSION_CHARS:
        raise ValueError(f"The {kind} is longer than {MAX_EXPRESSION_CHARS} characters")
    if "@" in text:
        raise ValueError(f"The {kind} cannot refer to variables with @; use column names and literal values")
    names: dict[str, str] = {}

    def placeholder(match: re.Match[str]) -> str:
        key = f"backticked_column_{len(names)}"
        names[key] = match.group(0)
        return key

    try:
        tree = ast.parse(_BACKTICKED.sub(placeholder, text), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"The {kind} is not valid: {exc.msg}") from exc

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            what = {
                ast.Attribute: "attribute or method access (.name)",
                ast.Subscript: "indexing ([...])",
                ast.Lambda: "lambda functions",
            }.get(type(node), type(node).__name__)
            raise ValueError(f"The {kind} cannot use {what}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError(f"The {kind} cannot use names starting with __")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                raise ValueError(f"The {kind} cannot use attribute or method access (.name)")
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
                allowed = ", ".join(FUNCTIONS)
                raise ValueError(f"The {kind} can only call these functions: {allowed}")
            if node.keywords:
                raise ValueError(f"The {kind} cannot pass keyword arguments to functions")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float, complex, str, bool, type(None))):
            raise ValueError(f"The {kind} contains an unsupported literal")
    return text
