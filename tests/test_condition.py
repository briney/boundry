"""Tests for boundry.condition module."""

from dataclasses import dataclass

import pytest

from boundry.condition import (
    ConditionError,
    check_condition,
    parse_condition,
    tokenize,
)


class TestTokenize:
    def test_simple_comparison(self):
        tokens = tokenize("{dG} < -10.0")
        kinds = [tok.kind for tok in tokens]
        assert kinds == ["VAR", "CMP", "OP", "NUMBER", "EOF"]

    def test_delta_function(self):
        tokens = tokenize("delta({final_energy}) < 0.5")
        assert tokens[0].kind == "FUNC"
        assert tokens[0].value == "delta"

    def test_dotted_path(self):
        tokens = tokenize("{sasa.buried_sasa} <= 10")
        assert tokens[0].kind == "VAR"
        assert tokens[0].value == "sasa.buried_sasa"

    def test_invalid_character(self):
        with pytest.raises(ConditionError, match="Invalid character"):
            tokenize("{dG} @ 1")

    def test_invalid_variable_path(self):
        with pytest.raises(ConditionError, match="Invalid variable path"):
            tokenize("{123bad} < 1")

    def test_unknown_function(self):
        with pytest.raises(ConditionError, match="Unknown function"):
            tokenize("sqrt({x}) < 1")

    def test_scientific_notation(self):
        tokens = tokenize("1e-5 < {x}")
        assert tokens[0].kind == "NUMBER"
        assert tokens[0].value == pytest.approx(1e-5)

    def test_two_char_operators(self):
        for op in ["<=", ">=", "==", "!="]:
            tokens = tokenize(f"{{x}} {op} 1")
            assert tokens[1].kind == "CMP"
            assert tokens[1].value == op


class TestParseCondition:
    def test_simple_less_than(self):
        ast = parse_condition("{dG} < -10")
        assert ast[0] == "cmp"
        assert ast[1] == "<"

    def test_arithmetic(self):
        ast = parse_condition("{a} + {b} < 10")
        assert ast[2][0] == "binop"
        assert ast[2][1] == "+"

    def test_function_call(self):
        ast = parse_condition("abs({dG}) > 5")
        assert ast[2][0] == "func"
        assert ast[2][1] == "abs"

    def test_delta(self):
        ast = parse_condition("delta({e}) < 0.5")
        assert ast[2][0] == "func"
        assert ast[2][1] == "delta"

    def test_missing_comparison_raises(self):
        with pytest.raises(ConditionError, match="Expected CMP"):
            parse_condition("{dG}")

    def test_parenthesized_expr(self):
        ast = parse_condition("({a} + {b}) * 2 < 10")
        assert ast[0] == "cmp"
        assert ast[2][0] == "binop"


@dataclass
class _Data:
    buried_sasa: float


class TestCheckCondition:
    def test_simple_true(self):
        assert check_condition("{dG} < -10.0", {"dG": -15.0}) is True

    def test_simple_false(self):
        assert check_condition("{dG} < -10.0", {"dG": -5.0}) is False

    def test_greater_than(self):
        assert check_condition("{x} > 3", {"x": 4}) is True

    def test_equality(self):
        assert check_condition("{x} == 3", {"x": 3}) is True

    def test_not_equal(self):
        assert check_condition("{x} != 3", {"x": 2}) is True

    def test_delta_with_previous(self):
        assert check_condition(
            "delta({e}) < 0.5",
            {"e": 10.2},
            previous_metadata={"e": 10.5},
        )

    def test_delta_without_previous_raises(self):
        with pytest.raises(ConditionError, match="requires previous"):
            check_condition("delta({e}) < 0.5", {"e": 10.2})

    def test_dotted_path(self):
        assert check_condition(
            "{sasa.buried_sasa} > 800",
            {"sasa": {"buried_sasa": 900}},
        )

    def test_missing_key_raises(self):
        with pytest.raises(ConditionError, match="Unknown metadata key"):
            check_condition("{missing} < 1", {"x": 1})

    def test_non_numeric_raises(self):
        with pytest.raises(ConditionError, match="Expected numeric"):
            check_condition("{name} < 1", {"name": "abc"})

    def test_abs_function(self):
        assert check_condition("abs({dG}) > 5", {"dG": -6})

    def test_arithmetic(self):
        assert check_condition("{a} + {b} * 2 == 7", {"a": 1, "b": 3})

    def test_division_by_zero(self):
        with pytest.raises(ConditionError, match="Division by zero"):
            check_condition("{x} / 0 < 1", {"x": 1})

    def test_dataclass_attribute_access(self):
        assert check_condition(
            "{sasa.buried_sasa} > 100",
            {"sasa": _Data(buried_sasa=300)},
        )
