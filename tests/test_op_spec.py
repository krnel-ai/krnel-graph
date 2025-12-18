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
