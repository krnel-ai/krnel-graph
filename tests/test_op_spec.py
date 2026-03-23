# Copyright (c) 2025-2026 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

from __future__ import annotations

from krnel.graph import OpSpec


class LeafSpec(OpSpec):
    value: str


class MixedSpec(OpSpec):
    leaf: LeafSpec
    optional_leaf: LeafSpec | None
    leaf_list: list[LeafSpec]
    leaf_map: dict[str, LeafSpec]
    description: str
    retries: int | None = None


def test_get_parameters_only_returns_non_opspec_fields():
    leaf = LeafSpec(value="foo")
    mixed = MixedSpec(
        leaf=leaf,
        optional_leaf=None,
        leaf_list=[leaf],
        leaf_map={"a": leaf},
        description="example",
        retries=3,
    )

    assert mixed.get_parameters() == {"description": "example", "retries": 3}


def test_get_parameters_includes_defaults():
    leaf = LeafSpec(value="bar")
    mixed = MixedSpec(
        leaf=leaf,
        optional_leaf=leaf,
        leaf_list=[leaf],
        leaf_map={"b": leaf},
        description="with-names",
    )

    assert mixed.get_parameters() == {"description": "with-names", "retries": None}


class NestedLiteralSpec(OpSpec):
    messages: list[dict[str, str]]
    config: dict[str, list[dict[str, int]]]


def test_code_repr_handles_nested_plain_literals():
    spec = NestedLiteralSpec(
        messages=[{"role": "system", "content": "hello"}],
        config={"limits": [{"max_tokens": 3}]},
    )

    code = spec.to_code(include_deps=False, include_banner_comment=False)

    assert "messages=[{'role': 'system', 'content': 'hello'}]," in code
    assert "config={'limits': [{'max_tokens': 3}]}," in code


def test_code_repr_handles_opspec_inside_nested_literals():
    leaf = LeafSpec(value="foo")
    mixed = MixedSpec(
        leaf=leaf,
        optional_leaf=None,
        leaf_list=[leaf],
        leaf_map={"a": leaf},
        description="example",
    )

    code = mixed.to_code(include_deps=False, include_banner_comment=False)

    assert f"leaf={leaf._code_repr_expr()}," in code
    assert f"leaf_list=[{leaf._code_repr_expr()}]," in code
    assert f"leaf_map={{'a': {leaf._code_repr_expr()}}}," in code
