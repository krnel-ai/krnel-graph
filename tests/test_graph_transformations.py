# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

# ruff: noqa: S101

import pytest
from pydantic import BaseModel, ConfigDict

from krnel.graph import OpSpec
from krnel.graph.graph_transformations import (
    get_dependencies,
    graph_substitute,
    map_fields,
)


class SimpleDataSource(OpSpec):
    """Simple data source for testing."""

    name: str


@pytest.fixture
def simple_source():
    return SimpleDataSource(name="test_source")


class ProcessingOp(OpSpec):
    """Processing operation for testing."""

    source: SimpleDataSource
    multiplier: int = 2


@pytest.fixture
def processing_op(simple_source):
    return ProcessingOp(source=simple_source, multiplier=3)


class CombineOp(OpSpec):
    """Operation that combines multiple sources."""

    sources: list[OpSpec]
    weights: dict[str, float] = {}


@pytest.fixture
def combine_op(simple_source):
    source2 = SimpleDataSource(name="source2")
    return CombineOp(sources=[simple_source, source2], weights={"a": 1.0, "b": 2.0})


class NestedOp(OpSpec):
    """Operation with nested dependencies."""

    primary: ProcessingOp
    secondary: SimpleDataSource
    backup_sources: list[SimpleDataSource] = []


@pytest.fixture
def nested_op(processing_op, simple_source):
    backup1 = SimpleDataSource(name="backup1")
    backup2 = SimpleDataSource(name="backup2")
    return NestedOp(
        primary=processing_op,
        secondary=simple_source,
        backup_sources=[backup1, backup2],
    )


class NonOpSpec(BaseModel):
    """Non-OpSpec model for testing type filtering."""

    model_config = ConfigDict(frozen=True)
    value: int


class MixedOp(OpSpec):
    """Operation with mixed OpSpec and non-OpSpec fields."""

    op_field: OpSpec
    non_op_field: NonOpSpec
    regular_field: str


class ChainableOp(OpSpec):
    """Operation that can chain to any other OpSpec."""

    source: OpSpec
    operation_type: str


# Tests for get_dependencies()


def test_get_dependencies_single_level(processing_op, simple_source):
    """Test getting dependencies from a single level operation."""
    deps = [
        op
        for (op, path) in get_dependencies(
            processing_op, filter_type=OpSpec, recursive=True
        )
    ]
    assert deps == [simple_source]


def test_get_dependencies_recursive(nested_op, processing_op, simple_source):
    """Test recursive dependency collection."""
    deps = [
        op
        for (op, path) in get_dependencies(
            nested_op, filter_type=OpSpec, recursive=True
        )
    ]
    # Should include all OpSpec instances except the root
    expected_deps = [processing_op, simple_source]
    # Also includes backup sources from nested_op
    backup_sources = nested_op.backup_sources
    expected_deps.extend(backup_sources)
    assert set(deps) == set(expected_deps)


def test_get_dependencies_no_recursive(nested_op, processing_op, simple_source):
    """Test non-recursive dependency collection."""
    deps = get_dependencies(nested_op, filter_type=OpSpec, recursive=False)
    deps = [op for (op, path) in deps]
    # Should only include direct dependencies
    expected_deps = [processing_op, simple_source]
    expected_deps.extend(nested_op.backup_sources)
    assert set(deps) == set(expected_deps)


def test_get_dependencies_type_filtering():
    """Test that type filtering works correctly."""
    non_op = NonOpSpec(value=42)
    simple_source = SimpleDataSource(name="test")
    mixed_op = MixedOp(
        op_field=simple_source, non_op_field=non_op, regular_field="test"
    )

    # Filter for OpSpec - should only get simple_source
    op_deps = get_dependencies(mixed_op, filter_type=OpSpec, recursive=True)
    op_deps = [op for (op, path) in op_deps]
    assert op_deps == [simple_source]

    # Filter for NonOpSpec - won't find non_op because get_dependencies only traverses
    # objects of the filter_type, and MixedOp is an OpSpec, not a NonOpSpec
    non_op_deps = get_dependencies(mixed_op, filter_type=NonOpSpec, recursive=True)
    non_op_deps = [op for (op, path) in non_op_deps]
    assert (
        non_op_deps == []
    )  # Empty because it doesn't traverse into OpSpec objects when looking for NonOpSpec
    # TODO(kwilber): ^ above behavior is almost certainly not what we want

    # But if we start with a NonOpSpec, it should find itself then exclude it
    direct_non_op_deps = get_dependencies(non_op, filter_type=NonOpSpec, recursive=True)
    direct_non_op_deps = [op for (op, path) in direct_non_op_deps]
    assert (
        direct_non_op_deps == []
    )  # Empty because non_op has no NonOpSpec dependencies


def test_get_dependencies_with_lists(combine_op):
    """Test dependency extraction from list fields."""
    deps = get_dependencies(combine_op, filter_type=OpSpec, recursive=True)
    deps = [op for (op, path) in deps]
    # Should include all sources from the list
    assert deps == combine_op.sources


def test_get_dependencies_excludes_self(simple_source):
    """Test that the root object excludes itself from dependencies."""
    deps = get_dependencies(simple_source, filter_type=OpSpec, recursive=True)
    deps = [op for (op, path) in deps]
    assert simple_source not in deps
    assert deps == []  # No dependencies for simple source


def test_get_dependencies_empty_when_no_matches():
    """Test that empty list is returned when no dependencies match the filter type."""
    simple_source = SimpleDataSource(name="test")
    deps = get_dependencies(
        simple_source, filter_type=NonOpSpec, recursive=True
    )  # No NonOpSpec dependencies
    deps = [op for (op, path) in deps]
    assert deps == []


# Tests for get_dependencies() with multiple roots


def test_get_dependencies_multiple_roots_basic():
    """Test get_dependencies with multiple root nodes."""
    source1 = SimpleDataSource(name="source1")
    source2 = SimpleDataSource(name="source2")
    processing1 = ProcessingOp(source=source1, multiplier=2)
    processing2 = ProcessingOp(source=source2, multiplier=3)

    # Get dependencies from both roots
    deps = get_dependencies(
        processing1, processing2, filter_type=OpSpec, recursive=True
    )
    deps = [op for (op, path) in deps]
    assert deps == [source1, source2]


def test_get_dependencies_multiple_roots_shared_dependencies():
    """Test get_dependencies with multiple roots that share dependencies."""
    shared_source = SimpleDataSource(name="shared")
    processing1 = ProcessingOp(source=shared_source, multiplier=2)
    processing2 = ProcessingOp(source=shared_source, multiplier=3)

    # Should get the shared dependency only once
    deps = get_dependencies(
        processing1, processing2, filter_type=OpSpec, recursive=True
    )
    deps = [op for (op, path) in deps]
    assert deps == [shared_source]


def test_get_dependencies_multiple_roots_complex():
    """Test get_dependencies with complex multiple roots graph."""
    source_a = SimpleDataSource(name="source_a")
    source_b = SimpleDataSource(name="source_b")
    source_c = SimpleDataSource(name="source_c")

    processing1 = ProcessingOp(source=source_a, multiplier=2)
    processing2 = ProcessingOp(source=source_b, multiplier=3)

    # Complex nested structure
    nested1 = NestedOp(
        primary=processing1, secondary=source_c, backup_sources=[source_a]
    )
    nested2 = NestedOp(
        primary=processing2, secondary=source_a, backup_sources=[source_b, source_c]
    )

    deps = get_dependencies(nested1, nested2, filter_type=OpSpec, recursive=True)
    deps = [op for (op, path) in deps]
    expected = [source_a, processing1, source_c, source_b, processing2]
    assert deps == expected


def test_get_dependencies_multiple_roots_exclude_roots():
    """Test that root nodes are only included if they are dependencies of other roots."""
    source1 = SimpleDataSource(name="source1")
    source2 = SimpleDataSource(name="source2")
    processing1 = ProcessingOp(source=source1, multiplier=2)
    processing2 = ProcessingOp(source=source2, multiplier=3)

    deps = get_dependencies(
        processing1, processing2, source1, filter_type=OpSpec, recursive=True
    )
    deps = [op for (op, path) in deps]
    # Should not include the roots themselves
    assert processing1 not in deps
    assert processing2 not in deps
    # Should include source1 because it's a dependency of processing1 (even though it's also a root)
    assert source1 in deps
    # Should still include source2 as it's a dependency of processing2
    assert source2 in deps


def test_get_dependencies_multiple_roots_non_recursive():
    """Test get_dependencies with multiple roots and recursive=False."""
    source_a = SimpleDataSource(name="source_a")
    source_b = SimpleDataSource(name="source_b")
    processing1 = ProcessingOp(source=source_a, multiplier=2)
    processing2 = ProcessingOp(source=source_b, multiplier=3)

    # Create nested ops that reference the processing ops
    nested1 = NestedOp(primary=processing1, secondary=source_b, backup_sources=[])
    nested2 = NestedOp(primary=processing2, secondary=source_a, backup_sources=[])

    # With recursive=False, should only get direct dependencies
    deps = get_dependencies(nested1, nested2, filter_type=OpSpec, recursive=False)
    deps = [op for (op, path) in deps]
    expected = [processing1, source_b, processing2, source_a]
    deps = get_dependencies(nested1, filter_type=OpSpec, recursive=False)
    deps = [op for (op, path) in deps]
    expected = [processing1, source_b]
    assert deps == expected


def test_get_dependencies_multiple_roots_empty_input():
    """Test get_dependencies with empty roots."""
    deps = get_dependencies(filter_type=OpSpec, recursive=True)
    deps = [op for (op, path) in deps]
    assert deps == []


def test_get_dependencies_multiple_roots_single_input():
    """Test that multiple roots API works with single input."""
    source = SimpleDataSource(name="source")
    processing = ProcessingOp(source=source, multiplier=2)

    # Should work the same as before
    deps = get_dependencies(processing, filter_type=OpSpec, recursive=True)
    deps = [op for (op, path) in deps]
    assert deps == [source]


# Tests for map_fields()


def test_map_fields_with_base_model():
    """Test mapping a BaseModel instance."""
    source = SimpleDataSource(name="original")

    def transform(obj, path):
        if isinstance(obj, SimpleDataSource):
            return SimpleDataSource(name=obj.name + "_transformed")
        return obj

    result = map_fields(source, SimpleDataSource, transform)
    assert isinstance(result, SimpleDataSource)
    assert result.name == "original_transformed"


def test_map_fields_with_list():
    """Test mapping items in a list."""
    source1 = SimpleDataSource(name="source1")
    source2 = SimpleDataSource(name="source2")
    source_list = [source1, source2]

    def transform(obj, path):
        if isinstance(obj, SimpleDataSource):
            return SimpleDataSource(name=obj.name + "_mapped")
        return obj

    result = map_fields(source_list, SimpleDataSource, transform)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0].name == "source1_mapped"
    assert result[1].name == "source2_mapped"


def test_map_fields_with_dict():
    """Test mapping values in a dictionary."""
    source1 = SimpleDataSource(name="source1")
    source2 = SimpleDataSource(name="source2")
    source_dict = {"a": source1, "b": source2}

    def transform(obj, path):
        assert path == ["a"] or path == ["b"]
        if isinstance(obj, SimpleDataSource):
            return SimpleDataSource(name=obj.name + "_mapped")
        return obj

    result = map_fields(source_dict, SimpleDataSource, transform)
    assert isinstance(result, dict)
    assert len(result) == 2
    assert result["a"].name == "source1_mapped"
    assert result["b"].name == "source2_mapped"


def test_map_fields_with_nested_structures():
    """Test mapping nested lists and dictionaries."""
    source = SimpleDataSource(name="source")
    nested_structure = {"list_field": [source], "dict_field": {"nested": source}}

    def transform(obj, path):
        assert path == ["list_field", 0] or path == ["dict_field", "nested"]
        if isinstance(obj, SimpleDataSource):
            return SimpleDataSource(name=obj.name + "_nested")
        return obj

    result = map_fields(nested_structure, SimpleDataSource, transform)
    assert isinstance(result, dict)
    assert result["list_field"][0].name == "source_nested"
    assert result["dict_field"]["nested"].name == "source_nested"


def test_map_fields_non_matching_types():
    """Test that non-matching types are returned unchanged."""
    mixed_data = {"string": "hello", "number": 42, "boolean": True, "none": None}

    def transform(obj, path):
        return obj  # This should never be called

    result = map_fields(mixed_data, SimpleDataSource, transform)
    assert result == mixed_data


def test_map_fields_empty_collections():
    """Test mapping empty lists and dictionaries."""

    def transform(obj, path):
        return obj

    empty_list_result = map_fields([], SimpleDataSource, transform)
    assert empty_list_result == []

    empty_dict_result = map_fields({}, SimpleDataSource, transform)
    assert empty_dict_result == {}


def test_map_fields_type_filtering():
    """Test that mapping only affects objects of the specified type."""
    source = SimpleDataSource(name="source")
    non_op = NonOpSpec(value=42)
    mixed_list = [source, non_op, "string", 123]

    def transform_source(obj, path):
        assert path == [0]
        return SimpleDataSource(name=obj.name + "_filtered")

    result = map_fields(mixed_list, SimpleDataSource, transform_source)
    assert len(result) == 4
    assert result[0].name == "source_filtered"
    assert result[1] == non_op  # unchanged
    assert result[2] == "string"  # unchanged
    assert result[3] == 123  # unchanged


# Tests for graph_substitute()


def test_graph_substitute_single_node():
    """Test substituting a single node in a simple graph."""
    source = SimpleDataSource(name="original")
    processing = ProcessingOp(source=source, multiplier=2)

    # Create replacement node
    new_source = SimpleDataSource(name="replacement")

    # Perform substitution
    new_roots = graph_substitute([processing], OpSpec, [(source, new_source)])

    assert len(new_roots) == 1
    new_processing = new_roots[0]
    assert isinstance(new_processing, ProcessingOp)
    assert new_processing.source == new_source
    assert new_processing.multiplier == 2


def test_graph_substitute_multiple_nodes():
    """Test substituting multiple nodes in a graph."""
    source1 = SimpleDataSource(name="source1")
    source2 = SimpleDataSource(name="source2")
    combine_op = CombineOp(sources=[source1, source2])

    # Create replacement nodes
    new_source1 = SimpleDataSource(name="new_source1")
    new_source2 = SimpleDataSource(name="new_source2")

    # Perform substitution
    new_roots = graph_substitute(
        [combine_op], OpSpec, [(source1, new_source1), (source2, new_source2)]
    )

    assert len(new_roots) == 1
    new_combine_op = new_roots[0]
    assert isinstance(new_combine_op, CombineOp)
    assert len(new_combine_op.sources) == 2
    assert new_source1 in new_combine_op.sources
    assert new_source2 in new_combine_op.sources


def test_graph_substitute_nested_dependencies():
    """Test substitution in a graph with nested dependencies."""
    source = SimpleDataSource(name="original_source")
    processing = ProcessingOp(source=source, multiplier=2)
    backup1 = SimpleDataSource(name="backup1")
    backup2 = SimpleDataSource(name="backup2")
    nested = NestedOp(
        primary=processing, secondary=source, backup_sources=[backup1, backup2]
    )

    # Replace the original source
    new_source = SimpleDataSource(name="new_source")

    new_roots = graph_substitute([nested], OpSpec, [(source, new_source)])

    assert len(new_roots) == 1
    new_nested = new_roots[0]
    assert isinstance(new_nested, NestedOp)

    # Check that the source was replaced in both places
    assert new_nested.secondary == new_source
    assert new_nested.primary.source == new_source

    # Backup sources should remain unchanged
    assert new_nested.backup_sources == [backup1, backup2]


def test_graph_substitute_multiple_roots():
    """Test substitution across multiple root nodes."""
    source = SimpleDataSource(name="shared_source")
    processing1 = ProcessingOp(source=source, multiplier=2)
    processing2 = ProcessingOp(source=source, multiplier=3)

    new_source = SimpleDataSource(name="new_shared_source")

    new_roots = graph_substitute(
        [processing1, processing2], OpSpec, [(source, new_source)]
    )

    assert len(new_roots) == 2
    assert all(root.source == new_source for root in new_roots)
    assert new_roots[0].multiplier == 2
    assert new_roots[1].multiplier == 3


def test_graph_substitute_no_substitutions():
    """Test graph_substitute with empty substitutions list."""
    source = SimpleDataSource(name="source")
    processing = ProcessingOp(source=source)

    new_roots = graph_substitute([processing], OpSpec, [])

    # Should return identical structure
    assert len(new_roots) == 1
    assert new_roots[0] == processing


def test_graph_substitute_node_not_in_graph():
    """Test that substitution fails when old node is not in the graph."""
    source = SimpleDataSource(name="source")
    processing = ProcessingOp(source=source)

    # Try to substitute a node that's not in the graph
    unrelated_node = SimpleDataSource(name="unrelated")
    new_node = SimpleDataSource(name="new")

    with pytest.raises(ValueError, match="Supposed to substitute"):
        graph_substitute([processing], OpSpec, [(unrelated_node, new_node)])


def test_graph_substitute_incomplete_substitutions():
    """Test that substitution fails when not all substitutions are made."""
    # This test checks the assertion that all intended substitutions were actually made
    # Since get_dependencies has a recursive parameter, we need to make sure our
    # substitution logic handles this correctly

    source1 = SimpleDataSource(name="source1")
    source2 = SimpleDataSource(name="source2")  # This won't be in the actual graph
    processing = ProcessingOp(source=source1)

    # Create substitutions where source2 is not actually in the graph dependencies
    new_source1 = SimpleDataSource(name="new_source1")
    new_source2 = SimpleDataSource(name="new_source2")

    # This should fail because source2 is not in the graph
    with pytest.raises(ValueError, match="Supposed to substitute"):
        graph_substitute(
            [processing], OpSpec, [(source1, new_source1), (source2, new_source2)]
        )


def test_graph_substitute_identity_substitution():
    """Test substitution where old and new nodes are different instances but equivalent."""
    source = SimpleDataSource(name="source")
    processing = ProcessingOp(source=source)

    # Create an equivalent but different instance
    equivalent_source = SimpleDataSource(name="source")

    new_roots = graph_substitute([processing], OpSpec, [(source, equivalent_source)])

    assert len(new_roots) == 1
    new_processing = new_roots[0]
    assert new_processing.source == equivalent_source
    # Even though they're equivalent, they should be different instances
    assert new_processing.source is equivalent_source
    assert new_processing.source is not source


def test_graph_substitute_preserves_non_targeted_types():
    """Test that substitution only affects the targeted filter type."""
    source = SimpleDataSource(name="source")
    non_op = NonOpSpec(value=42)
    mixed_op = MixedOp(op_field=source, non_op_field=non_op, regular_field="test")

    new_source = SimpleDataSource(name="new_source")

    # Only substitute OpSpec types
    new_roots = graph_substitute([mixed_op], OpSpec, [(source, new_source)])

    assert len(new_roots) == 1
    new_mixed_op = new_roots[0]
    assert isinstance(new_mixed_op, MixedOp)
    assert new_mixed_op.op_field == new_source
    assert new_mixed_op.non_op_field == non_op  # Should remain unchanged
    assert new_mixed_op.regular_field == "test"  # Should remain unchanged


# Tests for inter-root dependencies


def test_get_dependencies_inter_root_basic():
    """Test that inter-root dependencies are correctly included."""
    root_source = SimpleDataSource(name="root_source")
    root_processing = ProcessingOp(source=root_source, multiplier=2)

    # root_processing depends on root_source
    # root_source should appear in dependencies since it's a dependency of root_processing
    deps = get_dependencies(
        root_processing, root_source, filter_type=OpSpec, recursive=True
    )
    deps = [op for (op, path) in deps]
    assert root_source in deps
    assert root_processing not in deps  # root_processing shouldn't depend on itself


def test_get_dependencies_inter_root_chain():
    """Test inter-root dependency chains."""
    source_a = SimpleDataSource(name="source_a")
    source_b = SimpleDataSource(name="source_b")
    processing_b = ProcessingOp(source=source_b, multiplier=2)
    processing_a = ProcessingOp(source=source_a, multiplier=3)

    # Chain: processing_a -> source_a, processing_b -> source_b
    # No inter-root dependencies
    deps = get_dependencies(
        processing_a, processing_b, filter_type=OpSpec, recursive=True
    )
    deps = [op for (op, path) in deps]
    expected = [source_a, source_b]
    assert deps == expected
    assert processing_a not in deps
    assert processing_b not in deps


def test_get_dependencies_inter_root_complex_chain():
    """Test complex inter-root dependency chain."""
    base_source = SimpleDataSource(name="base")
    processing_1 = ProcessingOp(source=base_source, multiplier=2)
    processing_2 = ChainableOp(
        source=processing_1, operation_type="chain"
    )  # depends on processing_1

    # processing_2 -> processing_1 -> base_source
    # When both processing_2 and processing_1 are roots:
    # processing_1 should appear in deps because processing_2 depends on it
    deps = get_dependencies(
        processing_2, processing_1, filter_type=OpSpec, recursive=True
    )
    deps = [op for (op, path) in deps]
    expected = [base_source, processing_1]
    assert deps == expected
    assert (
        processing_2 not in deps
    )  # processing_2 is a root, shouldn't be in its own deps


def test_get_dependencies_inter_root_nested_structure():
    """Test inter-root dependencies in nested structures."""
    shared_source = SimpleDataSource(name="shared")
    backup_source = SimpleDataSource(name="backup")
    processing = ProcessingOp(source=shared_source, multiplier=2)

    # nested depends on both processing (root) and shared_source (also root)
    nested = NestedOp(
        primary=processing, secondary=shared_source, backup_sources=[backup_source]
    )

    deps = get_dependencies(
        nested, processing, shared_source, filter_type=OpSpec, recursive=True
    )
    deps = [op for (op, path) in deps]
    # Should include processing and shared_source since nested depends on them
    # Should also include backup_source since it's in the nested structure
    expected = [shared_source, processing, backup_source]
    assert deps == expected
    assert nested not in deps  # nested is a root


def test_get_dependencies_no_inter_root_dependencies():
    """Test case where roots are completely independent."""
    source_1 = SimpleDataSource(name="source1")
    source_2 = SimpleDataSource(name="source2")
    processing_1 = ProcessingOp(source=source_1, multiplier=2)
    processing_2 = ProcessingOp(source=source_2, multiplier=3)

    # Two independent processing chains
    deps = get_dependencies(
        processing_1, processing_2, filter_type=OpSpec, recursive=True
    )
    deps = [op for (op, path) in deps]
    expected = [source_1, source_2]
    assert deps == expected
    # Neither root should appear in dependencies
    assert processing_1 not in deps
    assert processing_2 not in deps


def test_get_dependencies_inter_root_with_shared_deps():
    """Test inter-root dependencies combined with shared dependencies."""
    shared_base = SimpleDataSource(name="shared_base")
    processing_1 = ProcessingOp(source=shared_base, multiplier=2)
    processing_2 = ChainableOp(
        source=processing_1, operation_type="chain"
    )  # depends on processing_1
    processing_3 = ProcessingOp(
        source=shared_base, multiplier=4
    )  # also depends on shared_base

    # processing_2 -> processing_1 -> shared_base
    # processing_3 -> shared_base
    # All three are roots
    deps = get_dependencies(
        processing_2, processing_1, processing_3, filter_type=OpSpec, recursive=True
    )
    deps = [op for (op, path) in deps]
    expected = [
        shared_base,
        processing_1,
    ]  # processing_1 is dep of processing_2, shared_base is dep of both
    assert deps == expected
    # None of the roots should be in their own dependency list
    assert processing_2 not in deps
    assert processing_3 not in deps
