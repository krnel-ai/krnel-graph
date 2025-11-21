"""Test that ephemeral operations correctly report has_result() based on dependencies."""

import pytest

from krnel.graph.dataset_ops import LoadInlineJsonDatasetOp, SelectColumnOp, TakeRowsOp
from krnel.graph.runners import LocalArrowRunner


@pytest.fixture
def runner(tmp_path):
    """Create a LocalArrowRunner with a temporary directory."""
    return LocalArrowRunner(str(tmp_path))


def test_ephemeral_op_without_completed_dependency(runner):
    """Test that an ephemeral operation returns False when its dependency hasn't been executed."""
    # Create a non-ephemeral operation (dataset load)
    dataset = LoadInlineJsonDatasetOp(
        data={
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [30, 25, 35],
        }
    )

    # Create an ephemeral operation that depends on the dataset
    select_op = SelectColumnOp(dataset=dataset, column_name="name")

    # The dataset hasn't been executed yet, so select_op should NOT be ready
    assert not select_op.has_result(runner=runner), (
        "SelectColumnOp.has_result() should return False when its dependency "
        "(LoadInlineJsonDatasetOp) hasn't been executed yet"
    )


def test_ephemeral_op_with_completed_dependency(runner):
    """Test that an ephemeral operation returns True when its dependency has been executed."""
    # Create a non-ephemeral operation (dataset load)
    dataset = LoadInlineJsonDatasetOp(
        data={
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [30, 25, 35],
        }
    )

    # Execute the dataset operation to cache its result
    runner.to_arrow(dataset)

    # Now the dataset has been executed, so it should have a result
    assert dataset.has_result(runner=runner), "Dataset should have a result after execution"

    # Create an ephemeral operation that depends on the dataset
    select_op = SelectColumnOp(dataset=dataset, column_name="name")

    # Now that the dependency is complete, select_op should be ready
    assert select_op.has_result(runner=runner), (
        "SelectColumnOp.has_result() should return True when its dependency "
        "(LoadInlineJsonDatasetOp) has been executed"
    )


def test_nested_ephemeral_ops(runner):
    """Test ephemeral operations that depend on other ephemeral operations."""
    # Create a non-ephemeral operation (dataset load)
    dataset = LoadInlineJsonDatasetOp(
        data={
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [30, 25, 35],
        }
    )

    # Create first ephemeral operation (take rows)
    take_first = TakeRowsOp(dataset=dataset, num_rows=2)

    # Create second ephemeral operation that depends on the first (take more rows)
    take_second = TakeRowsOp(dataset=take_first, num_rows=1)

    # Neither should be ready because the base dataset hasn't been executed
    assert not take_first.has_result(runner=runner), (
        "First ephemeral op should not be ready when base dataset is not executed"
    )
    assert not take_second.has_result(runner=runner), (
        "Second ephemeral op should not be ready when base dataset is not executed"
    )

    # Execute the base dataset
    runner.to_arrow(dataset)

    # Now both ephemeral operations should be ready
    assert take_first.has_result(runner=runner), (
        "First ephemeral op should be ready after base dataset is executed"
    )
    assert take_second.has_result(runner=runner), (
        "Second ephemeral op should be ready after base dataset is executed"
    )


def test_ephemeral_op_with_multiple_dependencies(runner):
    """Test ephemeral operation with multiple dependencies, some complete and some incomplete."""
    # Create two independent datasets
    dataset1 = LoadInlineJsonDatasetOp(
        data={
            "id": [1, 2],
            "value": [10, 20],
        }
    )

    dataset2 = LoadInlineJsonDatasetOp(
        data={
            "id": [3, 4],
            "value": [30, 40],
        }
    )

    # Execute only dataset1
    runner.to_arrow(dataset1)

    # Verify dataset1 has result but dataset2 doesn't
    assert dataset1.has_result(runner=runner), "dataset1 should have a result"
    assert not dataset2.has_result(runner=runner), "dataset2 should not have a result"

    # Create ephemeral operations from each dataset
    select1 = SelectColumnOp(dataset=dataset1, column_name="value")
    select2 = SelectColumnOp(dataset=dataset2, column_name="value")

    # select1 should be ready (its dependency is complete)
    assert select1.has_result(runner=runner), (
        "select1 should be ready because dataset1 is complete"
    )

    # select2 should NOT be ready (its dependency is incomplete)
    assert not select2.has_result(runner=runner), (
        "select2 should not be ready because dataset2 is incomplete"
    )

    # Now execute dataset2
    runner.to_arrow(dataset2)

    # Now select2 should be ready
    assert select2.has_result(runner=runner), (
        "select2 should be ready after dataset2 is executed"
    )


def test_ephemeral_op_is_actually_ephemeral(runner):
    """Sanity check: verify that the operations we're testing are actually ephemeral."""
    dataset = LoadInlineJsonDatasetOp(data={"id": [1]})
    select_op = SelectColumnOp(dataset=dataset, column_name="id")
    take_op = TakeRowsOp(dataset=dataset, num_rows=1)

    # These should be ephemeral
    assert select_op.is_ephemeral, "SelectColumnOp should be ephemeral"
    assert take_op.is_ephemeral, "TakeRowsOp should be ephemeral"

    # The base dataset should NOT be ephemeral
    assert not dataset.is_ephemeral, "LoadInlineJsonDatasetOp should not be ephemeral"
