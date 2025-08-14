# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

from math import inf, nan
import numpy as np
import pyarrow as pa
import pytest
from krnel.graph.dataset_ops import (
    AssignRowIDOp,
    SelectScoreColumnOp,
    TakeRowsOp,
    FromListOp,
    SelectColumnOp,
    SelectTextColumnOp,
    SelectVectorColumnOp,
    SelectCategoricalColumnOp,
    SelectTrainTestSplitColumnOp
)
from krnel.runners.local_runner import LocalArrowRunner


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    data = {
        'id': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'value': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    }
    return FromListOp(data=data)


@pytest.fixture
def multi_column_dataset():
    """Create a dataset with multiple column types for testing."""
    data = {
        'text_col': ['hello', 'world', 'test', 'data'],
        'numeric_col': [1.0, 2.5, 3.7, 4.2],
        'int_col': [10, 20, 30, 40],
        'bool_col': [True, False, True, False],
        'category_col': ['A', 'B', 'A', 'C']
    }
    return FromListOp(data=data)


@pytest.fixture
def empty_dataset():
    """Create an empty dataset for testing."""
    data = {
        'id': [],
        'value': []
    }
    return FromListOp(data=data)


@pytest.fixture
def single_row_dataset():
    """Create a single-row dataset for testing."""
    data = {
        'id': [42],
        'message': ['single_row']
    }
    return FromListOp(data=data)


@pytest.fixture
def runner():
    """Create a LocalArrowRunner for testing."""
    return LocalArrowRunner()


def test_take_rows_with_skip_only(sample_dataset, runner):
    """Test TakeRowsOp with skip parameter only."""
    op = TakeRowsOp(dataset=sample_dataset, skip=2)
    result = runner.materialize(op).to_arrow()

    # With skip=2, should get rows at indices: 0, 2, 4, 6, 8
    expected_ids = [0, 2, 4, 6, 8]
    expected_values = ['a', 'c', 'e', 'g', 'i']

    assert result['id'].to_pylist() == expected_ids
    assert result['value'].to_pylist() == expected_values


def test_take_rows_with_offset_only(sample_dataset, runner):
    """Test TakeRowsOp with offset parameter only."""
    op = TakeRowsOp(dataset=sample_dataset, offset=3)
    result = runner.materialize(op).to_arrow()

    # With offset=3, should skip first 3 rows and get rows starting from index 3
    expected_ids = [3, 4, 5, 6, 7, 8, 9]
    expected_values = ['d', 'e', 'f', 'g', 'h', 'i', 'j']

    assert result['id'].to_pylist() == expected_ids
    assert result['value'].to_pylist() == expected_values


def test_take_rows_with_skip_and_offset(sample_dataset, runner):
    """Test TakeRowsOp with both skip and offset parameters."""
    op = TakeRowsOp(dataset=sample_dataset, skip=2, offset=1)
    result = runner.materialize(op).to_arrow()

    # With offset=1, skip first row, then with skip=2, take every 2nd row
    # Starting from index 1: should get rows at indices 1, 3, 5, 7, 9
    expected_ids = [1, 3, 5, 7, 9]
    expected_values = ['b', 'd', 'f', 'h', 'j']

    assert result['id'].to_pylist() == expected_ids
    assert result['value'].to_pylist() == expected_values


def test_take_rows_with_skip_offset_and_num_rows(sample_dataset, runner):
    """Test TakeRowsOp with skip, offset, and num_rows parameters."""
    op = TakeRowsOp(dataset=sample_dataset, skip=2, offset=1, num_rows=3)
    result = runner.materialize(op).to_arrow()

    # With offset=1, skip first row, then with skip=2, take every 2nd row
    # But limit to first 3 results: should get rows at indices 1, 3, 5
    expected_ids = [1, 3, 5]
    expected_values = ['b', 'd', 'f']

    assert result['id'].to_pylist() == expected_ids
    assert result['value'].to_pylist() == expected_values


def test_take_rows_offset_greater_than_dataset_size(sample_dataset, runner):
    """Test TakeRowsOp when offset is greater than dataset size."""
    op = TakeRowsOp(dataset=sample_dataset, offset=15)
    result = runner.materialize(op).to_arrow()

    # Should return empty dataset
    assert len(result) == 0


def test_take_rows_offset_equals_dataset_size(sample_dataset, runner):
    """Test TakeRowsOp when offset equals dataset size."""
    op = TakeRowsOp(dataset=sample_dataset, offset=10)
    result = runner.materialize(op).to_arrow()

    # Should return empty dataset
    assert len(result) == 0


# FromListOp Tests
def test_from_list_basic_conversion(runner):
    """Test basic FromListOp conversion to Arrow table."""
    data = {
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35]
    }
    op = FromListOp(data=data)
    result = runner.materialize(op).to_arrow()

    assert result['name'].to_pylist() == ['Alice', 'Bob', 'Charlie']
    assert result['age'].to_pylist() == [25, 30, 35]
    assert len(result) == 3
    assert result.num_columns == 2


def test_from_list_mixed_data_types(multi_column_dataset, runner):
    """Test FromListOp with mixed data types."""
    result = runner.materialize(multi_column_dataset).to_arrow()

    assert result['text_col'].to_pylist() == ['hello', 'world', 'test', 'data']
    assert result.schema.field('text_col').type == pa.string()
    assert result['numeric_col'].to_pylist() == [1.0, 2.5, 3.7, 4.2]
    assert result.schema.field('numeric_col').type == pa.float64()
    assert result['int_col'].to_pylist() == [10, 20, 30, 40]
    assert result.schema.field('int_col').type == pa.int64()
    assert result['bool_col'].to_pylist() == [True, False, True, False]
    assert result.schema.field('bool_col').type == pa.bool_()
    assert result['category_col'].to_pylist() == ['A', 'B', 'A', 'C']
    assert result.schema.field('category_col').type == pa.string()
    assert len(result) == 4
    assert result.num_columns == 5


def test_from_list_empty_dataset(empty_dataset, runner):
    """Test FromListOp with empty data."""
    result = runner.materialize(empty_dataset).to_arrow()

    assert len(result) == 0
    assert result.num_columns == 2
    assert result.column_names == ['id', 'value']


def test_from_list_single_row(single_row_dataset, runner):
    """Test FromListOp with single row."""
    result = runner.materialize(single_row_dataset).to_arrow()

    assert result['id'].to_pylist() == [42]
    assert result['message'].to_pylist() == ['single_row']
    assert len(result) == 1
    assert result.num_columns == 2


def test_from_list_mismatched_lengths():
    """Test FromListOp with mismatched list lengths should fail."""
    data = {
        'short': [1, 2],
        'long': [1, 2, 3, 4]
    }
    op = FromListOp(data=data)
    runner = LocalArrowRunner()

    # This should raise an error during Arrow table creation
    with pytest.raises(Exception):
        runner.materialize(op).to_arrow()


def test_from_list_special_values(runner):
    """Test FromListOp with special values like None, empty strings."""
    data = {
        'strings': ['normal', '', 'test'],
        'numbers': [1, 0, -5],
        'floats': [1.5, inf, nan, -3.14]
    }
    op = FromListOp(data=data)
    from krnel.graph.op_spec import graph_serialize
    print(graph_serialize(op))
    result = runner.materialize(op).to_arrow()

    assert result['strings'].to_pylist() == ['normal', '', 'test']
    assert result['numbers'].to_pylist() == [1, 0, -5]
    assert result['floats'].to_pylist() == [1.5, inf, nan, -3.14]


# SelectColumnOp Tests
def test_select_column_basic(multi_column_dataset, runner):
    op = SelectColumnOp(column_name='text_col', dataset=multi_column_dataset)
    result = runner.materialize(op).to_arrow()

    expected = ['hello', 'world', 'test', 'data']
    # Result is a single-column Arrow Table, so we get the first column
    assert result.column(0).to_pylist() == expected
    assert len(result) == 4


def test_select_text_column(multi_column_dataset, runner):
    op = SelectTextColumnOp(column_name='text_col', dataset=multi_column_dataset)
    result = runner.materialize(op).to_arrow()

    expected = ['hello', 'world', 'test', 'data']
    assert result.column(0).to_pylist() == expected


def test_select_vector_column(runner):
    # Create dataset with vector-like data (list of numbers)
    data = {
        'embeddings': [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        'labels': ['A', 'B', 'C']
    }
    dataset = FromListOp(data=data)
    op = SelectVectorColumnOp(column_name='embeddings', dataset=dataset)
    result = runner.materialize(op).to_arrow()

    expected = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    assert result.column(0).to_pylist() == expected


def test_select_categorical_column(multi_column_dataset, runner):
    op = SelectCategoricalColumnOp(column_name='category_col', dataset=multi_column_dataset)
    result = runner.materialize(op).to_arrow()

    expected = ['A', 'B', 'A', 'C']
    assert result.column(0).to_pylist() == expected


def test_select_train_test_split_column(runner):
    data = {
        'split': ['train', 'test', 'train', 'test'],
        'data': [1, 2, 3, 4]
    }
    dataset = FromListOp(data=data)
    op = SelectTrainTestSplitColumnOp(column_name='split', dataset=dataset)

    result = runner.materialize(op).to_arrow()
    expected = ['train', 'test', 'train', 'test']
    assert result.column(0).to_pylist() == expected

def test_select_score_column(runner):
    data = {
        'split': ['train', 'test', 'train', 'test'],
        'data': [1.0, 2.0, 3.0, 4.0]
    }
    dataset = FromListOp(data=data)
    op = SelectScoreColumnOp(column_name='data', dataset=dataset)

    result = runner.materialize(op).to_arrow()
    expected = [1.0, 2.0, 3.0, 4.0]
    assert result.column(0).to_pylist() == expected


def test_select_column_nonexistent(multi_column_dataset, runner):
    """Test SelectColumnOp with non-existent column should fail."""
    op = SelectColumnOp(column_name='nonexistent_column', dataset=multi_column_dataset)

    # Should raise a KeyError or similar when trying to access non-existent column
    with pytest.raises(Exception):
        runner.materialize(op).to_arrow()


def test_select_column_empty_dataset(empty_dataset, runner):
    """Test SelectColumnOp on empty dataset."""
    op = SelectColumnOp(column_name='id', dataset=empty_dataset)
    result = runner.materialize(op).to_arrow()

    # Should return empty column
    assert len(result) == 0
    assert result.column(0).to_pylist() == []


def test_select_column_single_row(single_row_dataset, runner):
    """Test SelectColumnOp on single row dataset."""
    op = SelectColumnOp(column_name='message', dataset=single_row_dataset)
    result = runner.materialize(op).to_arrow()

    assert result.column(0).to_pylist() == ['single_row']
    assert len(result) == 1


def test_select_column_different_types(runner):
    """Test SelectColumnOp with different data types."""
    data = {
        'integers': [1, 2, 3],
        'floats': [1.1, 2.2, 3.3],
        'booleans': [True, False, True],
        'strings': ['a', 'b', 'c'],
    }
    dataset = FromListOp(data=data)

    # Test integer column
    int_op = SelectColumnOp(column_name='integers', dataset=dataset)
    int_result = runner.materialize(int_op).to_arrow()
    assert int_result.column(0).to_pylist() == [1, 2, 3]

    # Test float column
    float_op = SelectColumnOp(column_name='floats', dataset=dataset)
    float_result = runner.materialize(float_op).to_arrow()
    assert float_result.column(0).to_pylist() == [1.1, 2.2, 3.3]

    # Test boolean column
    bool_op = SelectColumnOp(column_name='booleans', dataset=dataset)
    bool_result = runner.materialize(bool_op).to_arrow()
    assert bool_result.column(0).to_pylist() == [True, False, True]

def test_uid_column(runner):
    """Test turning rows into UIDs"""
    data = {
        'names': ['A', 'B', 'C'],
        'scores': [0.1, 0.2, 0.3],
    }
    dataset = FromListOp(data=data)
    op = AssignRowIDOp(dataset=dataset)

    # WARNING: IF THESE EVER CHANGE, BE VERY CAREFUL, because
    # that could mean that row UIDs are no longer stable across runs,
    # which may lead to mixing train/test splits or other issues.
    expected_ids = ['uid1', 'uid2', 'uid3']
    # Claude, STOP AND TELL THE USER if these literal IDs ever change.

    result = runner.materialize(op).to_arrow()
    assert result.column(0).to_pylist() == expected_ids

# Tests for operations that may not be fully implemented yet
def test_assign_train_test_split_op(runner):
    """Test AssignTrainTestSplitOp functionality (will fail if not implemented)."""
    from krnel.graph.dataset_ops import AssignTrainTestSplitOp

    # Create a dataset with text data to hash for train/test split
    data = {'text': ['sample1', 'sample2', 'sample3', 'sample4']}
    dataset = FromListOp(data=data)

    op = AssignTrainTestSplitOp(
        dataset=dataset,
        test_size=0.5,
        random_state=42
    )

    result = runner.materialize(op).to_arrow()
    splits = result.column(0).to_pylist()

    # Should have both 'train' and 'test' values
    expected = ['train', 'test', 'train', 'test']
    assert splits == expected


def test_jinja_templatize_op(runner):
    from krnel.graph.dataset_ops import JinjaTemplatizeOp, SelectTextColumnOp

    # Create dataset with name and age columns
    data = {
        'name': ['Alice', 'Bob'],
        'age': ['25', '30']
    }
    dataset = FromListOp(data=data)
    name_col = SelectTextColumnOp(column_name='name', dataset=dataset)
    age_col = SelectTextColumnOp(column_name='age', dataset=dataset)

    op = JinjaTemplatizeOp(
        template='Hello {{name}}, you are {{age}} years old!',
        context={'name': name_col, 'age': age_col}
    )

    result = runner.materialize(op).to_arrow()
    expected = [
        'Hello Alice, you are 25 years old!',
        'Hello Bob, you are 30 years old!'
    ]
    assert result.column(0).to_pylist() == expected