"""Safe parser/evaluator for workflow conditions."""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

_NUMERIC_RE = re.compile(
    r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
)
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SEGMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_COMPARE_OPS = {"<", ">", "<=", ">=", "==", "!="}
_MATH_OPS = {"+", "-", "*", "/"}
_FUNCTIONS = {"abs", "delta"}

ASTNode = Union[
    Tuple[str, float],
    Tuple[str, str],
    Tuple[str, str, Any, Any],
    Tuple[str, str, Any],
]


class ConditionError(Exception):
    """Raised for invalid condition strings or evaluation failures."""


@dataclass(frozen=True)
class _Token:
    kind: str
    value: Any
    position: int


def tokenize(expr: str) -> List[_Token]:
    """Tokenize a condition expression."""
    tokens: List[_Token] = []
    i = 0
    length = len(expr)

    while i < length:
        ch = expr[i]
        if ch.isspace():
            i += 1
            continue

        if i + 1 < length and expr[i : i + 2] in _COMPARE_OPS:
            tokens.append(_Token("CMP", expr[i : i + 2], i))
            i += 2
            continue

        if ch in "<>":
            tokens.append(_Token("CMP", ch, i))
            i += 1
            continue

        if ch in _MATH_OPS:
            tokens.append(_Token("OP", ch, i))
            i += 1
            continue

        if ch == "(":
            tokens.append(_Token("LPAREN", ch, i))
            i += 1
            continue
        if ch == ")":
            tokens.append(_Token("RPAREN", ch, i))
            i += 1
            continue

        if ch == "{":
            end = expr.find("}", i + 1)
            if end == -1:
                raise ConditionError(
                    f"Unterminated variable reference at position {i}"
                )
            raw_path = expr[i + 1 : end]
            _validate_variable_path(raw_path)
            tokens.append(_Token("VAR", raw_path, i))
            i = end + 1
            continue

        match = _NUMERIC_RE.match(expr, i)
        if match:
            value = float(match.group(0))
            tokens.append(_Token("NUMBER", value, i))
            i = match.end()
            continue

        ident_match = _IDENT_RE.match(expr, i)
        if ident_match:
            name = ident_match.group(0)
            if name not in _FUNCTIONS:
                raise ConditionError(
                    f"Unknown function '{name}' at position {i}"
                )
            tokens.append(_Token("FUNC", name, i))
            i = ident_match.end()
            continue

        raise ConditionError(
            f"Invalid character '{ch}' at position {i}"
        )

    tokens.append(_Token("EOF", "", len(expr)))
    return tokens


@lru_cache(maxsize=512)
def parse_condition(condition_str: str) -> ASTNode:
    """Parse a condition string into an AST."""
    parser = _Parser(tokenize(condition_str))
    return parser.parse_condition()


def check_condition(
    condition_str: str,
    metadata: Dict[str, Any],
    previous_metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Parse and evaluate a condition string."""
    ast = parse_condition(condition_str)
    result = _evaluate_node(ast, metadata, previous_metadata)
    if not isinstance(result, bool):
        raise ConditionError("Condition did not evaluate to boolean")
    return result


class _Parser:
    def __init__(self, tokens: List[_Token]):
        self.tokens = tokens
        self.index = 0

    def parse_condition(self) -> ASTNode:
        left = self._parse_expr()
        cmp_tok = self._expect("CMP")
        right = self._parse_expr()
        self._expect("EOF")
        return ("cmp", cmp_tok.value, left, right)

    def _parse_expr(self) -> ASTNode:
        node = self._parse_term()
        while self._match("OP", "+") or self._match("OP", "-"):
            op = self._advance().value
            rhs = self._parse_term()
            node = ("binop", op, node, rhs)
        return node

    def _parse_term(self) -> ASTNode:
        node = self._parse_factor()
        while self._match("OP", "*") or self._match("OP", "/"):
            op = self._advance().value
            rhs = self._parse_factor()
            node = ("binop", op, node, rhs)
        return node

    def _parse_factor(self) -> ASTNode:
        if self._match("NUMBER"):
            return ("num", self._advance().value)
        if self._match("VAR"):
            return ("var", self._advance().value)
        if self._match("FUNC"):
            func = self._advance().value
            self._expect("LPAREN")
            arg = self._parse_expr()
            self._expect("RPAREN")
            return ("func", func, arg)
        if self._match("LPAREN"):
            self._advance()
            inner = self._parse_expr()
            self._expect("RPAREN")
            return inner
        if self._match("OP", "-"):
            self._advance()
            inner = self._parse_factor()
            return ("binop", "*", ("num", -1.0), inner)
        token = self._peek()
        raise ConditionError(
            f"Unexpected token '{token.kind}' at position {token.position}"
        )

    def _peek(self) -> _Token:
        return self.tokens[self.index]

    def _advance(self) -> _Token:
        token = self.tokens[self.index]
        self.index += 1
        return token

    def _match(self, kind: str, value: Optional[str] = None) -> bool:
        token = self._peek()
        if token.kind != kind:
            return False
        if value is not None and token.value != value:
            return False
        return True

    def _expect(self, kind: str, value: Optional[str] = None) -> _Token:
        if not self._match(kind, value):
            token = self._peek()
            expected = (
                f"{kind}({value})" if value is not None else kind
            )
            found = f"{token.kind}({token.value})"
            raise ConditionError(
                f"Expected {expected}, found {found} "
                f"at position {token.position}"
            )
        return self._advance()


def _evaluate_node(
    node: ASTNode,
    metadata: Dict[str, Any],
    previous_metadata: Optional[Dict[str, Any]],
    in_delta: bool = False,
) -> Union[float, bool]:
    tag = node[0]
    if tag == "cmp":
        _, op, left, right = node
        left_value = _to_float(
            _evaluate_node(left, metadata, previous_metadata, in_delta)
        )
        right_value = _to_float(
            _evaluate_node(right, metadata, previous_metadata, in_delta)
        )
        return _compare(op, left_value, right_value)

    if tag == "num":
        return float(node[1])

    if tag == "var":
        value = _resolve_var(node[1], metadata)
        return _to_float(value)

    if tag == "binop":
        _, op, left, right = node
        left_value = _to_float(
            _evaluate_node(left, metadata, previous_metadata, in_delta)
        )
        right_value = _to_float(
            _evaluate_node(right, metadata, previous_metadata, in_delta)
        )
        return _apply_binop(op, left_value, right_value)

    if tag == "func":
        _, name, arg = node
        if name == "abs":
            value = _to_float(
                _evaluate_node(arg, metadata, previous_metadata, in_delta)
            )
            return abs(value)
        if name == "delta":
            if in_delta:
                raise ConditionError("Nested delta() is not supported")
            if previous_metadata is None:
                raise ConditionError(
                    "delta() requires previous metadata"
                )
            current_value = _to_float(
                _evaluate_node(arg, metadata, previous_metadata, True)
            )
            prev_value = _to_float(
                _evaluate_node(arg, previous_metadata, None, True)
            )
            return abs(current_value - prev_value)
        raise ConditionError(f"Unsupported function '{name}'")

    raise ConditionError(f"Unsupported AST node '{tag}'")


def _resolve_var(path: str, metadata: Dict[str, Any]) -> Any:
    current: Any = metadata
    for segment in path.split("."):
        if segment.startswith("_"):
            raise ConditionError(
                f"Private path segments are not allowed: '{path}'"
            )

        if isinstance(current, dict):
            if segment not in current:
                raise ConditionError(f"Unknown metadata key '{path}'")
            current = current[segment]
            continue

        if dataclasses.is_dataclass(current):
            if not hasattr(current, segment):
                raise ConditionError(f"Unknown metadata key '{path}'")
            current = getattr(current, segment)
            continue

        if hasattr(current, segment) and not segment.startswith("_"):
            current = getattr(current, segment)
            continue

        raise ConditionError(f"Unknown metadata key '{path}'")

    return current


def _to_float(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConditionError(
            f"Expected numeric value, got {type(value).__name__}"
        )
    return float(value)


def _apply_binop(op: str, left: float, right: float) -> float:
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op == "*":
        return left * right
    if op == "/":
        if right == 0:
            raise ConditionError("Division by zero")
        return left / right
    raise ConditionError(f"Unsupported operator '{op}'")


def _compare(op: str, left: float, right: float) -> bool:
    if op == "<":
        return left < right
    if op == ">":
        return left > right
    if op == "<=":
        return left <= right
    if op == ">=":
        return left >= right
    if op == "==":
        return left == right
    if op == "!=":
        return left != right
    raise ConditionError(f"Unsupported comparator '{op}'")


def _validate_variable_path(path: str) -> None:
    if not path:
        raise ConditionError("Empty variable path '{}'")
    segments = path.split(".")
    for segment in segments:
        if not _SEGMENT_RE.fullmatch(segment):
            raise ConditionError(
                f"Invalid variable path '{{{path}}}'"
            )
        if segment.startswith("_"):
            raise ConditionError(
                f"Private variable path '{{{path}}}' is not allowed"
            )
