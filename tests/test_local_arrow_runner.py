# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

# ruff: noqa: S101, B017

from math import inf, nan

import numpy as np
import pyarrow as pa
import pytest

from krnel.graph.dataset_ops import (
    AssignRowIDOp,
    AssignTrainTestSplitOp,
    CategoryToBooleanOp,
    JinjaTemplatizeOp,
    LoadInlineJsonDatasetOp,
    MaskRowsOp,
    SelectBooleanColumnOp,
    SelectCategoricalColumnOp,
    SelectColumnOp,
    SelectScoreColumnOp,
    SelectTextColumnOp,
    SelectTrainTestSplitColumnOp,
    SelectVectorColumnOp,
    TakeRowsOp,
)
from krnel.graph.runners.local_runner import LocalArrowRunner


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    data = {
        "id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "value": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
    }
    return LoadInlineJsonDatasetOp(data=data)


@pytest.fixture
def multi_column_dataset():
    """Create a dataset with multiple column types for testing."""
    data = {
        "text_col": ["hello", "world", "test", "data"],
        "numeric_col": [1.0, 2.5, 3.7, 4.2],
        "int_col": [10, 20, 30, 40],
        "bool_col": [True, False, True, False],
        "category_col": ["A", "B", "A", "C"],
    }
    return LoadInlineJsonDatasetOp(data=data)


@pytest.fixture
def empty_dataset():
    """Create an empty dataset for testing."""
    data = {"id": [], "value": []}
    return LoadInlineJsonDatasetOp(data=data)


@pytest.fixture
def single_row_dataset():
    """Create a single-row dataset for testing."""
    data = {"id": [42], "message": ["single_row"]}
    return LoadInlineJsonDatasetOp(data=data)


@pytest.fixture
def runner():
    """Create a LocalArrowRunner for testing."""
    return LocalArrowRunner(store_uri="memory://")


def test_take_rows_with_skip_only(sample_dataset, runner):
    """Test TakeRowsOp with skip parameter only."""
    op = TakeRowsOp(dataset=sample_dataset, skip=2)
    result = runner.to_arrow(op)

    # With skip=2, should get rows at indices: 0, 2, 4, 6, 8
    expected_ids = [0, 2, 4, 6, 8]
    expected_values = ["a", "c", "e", "g", "i"]

    assert result["id"].to_pylist() == expected_ids
    assert result["value"].to_pylist() == expected_values


def test_take_rows_with_offset_only(sample_dataset, runner):
    """Test TakeRowsOp with offset parameter only."""
    op = TakeRowsOp(dataset=sample_dataset, offset=3)
    result = runner.to_arrow(op)

    # With offset=3, should skip first 3 rows and get rows starting from index 3
    expected_ids = [3, 4, 5, 6, 7, 8, 9]
    expected_values = ["d", "e", "f", "g", "h", "i", "j"]

    assert result["id"].to_pylist() == expected_ids
    assert result["value"].to_pylist() == expected_values


def test_take_rows_with_skip_and_offset(sample_dataset, runner):
    """Test TakeRowsOp with both skip and offset parameters."""
    op = TakeRowsOp(dataset=sample_dataset, skip=2, offset=1)
    result = runner.to_arrow(op)

    # With offset=1, skip first row, then with skip=2, take every 2nd row
    # Starting from index 1: should get rows at indices 1, 3, 5, 7, 9
    expected_ids = [1, 3, 5, 7, 9]
    expected_values = ["b", "d", "f", "h", "j"]

    assert result["id"].to_pylist() == expected_ids
    assert result["value"].to_pylist() == expected_values


def test_take_rows_with_skip_offset_and_num_rows(sample_dataset, runner):
    """Test TakeRowsOp with skip, offset, and num_rows parameters."""
    op = TakeRowsOp(dataset=sample_dataset, skip=2, offset=1, num_rows=3)
    result = runner.to_arrow(op)

    # With offset=1, skip first row, then with skip=2, take every 2nd row
    # But limit to first 3 results: should get rows at indices 1, 3, 5
    expected_ids = [1, 3, 5]
    expected_values = ["b", "d", "f"]

    assert result["id"].to_pylist() == expected_ids
    assert result["value"].to_pylist() == expected_values


def test_take_rows_offset_greater_than_dataset_size(sample_dataset, runner):
    """Test TakeRowsOp when offset is greater than dataset size."""
    op = TakeRowsOp(dataset=sample_dataset, offset=15)
    result = runner.to_arrow(op)

    # Should return empty dataset
    assert len(result) == 0


def test_take_rows_offset_equals_dataset_size(sample_dataset, runner):
    """Test TakeRowsOp when offset equals dataset size."""
    op = TakeRowsOp(dataset=sample_dataset, offset=10)
    result = runner.to_arrow(op)

    # Should return empty dataset
    assert len(result) == 0


# LoadInlineJsonDatasetOp Tests
def test_from_list_basic_conversion(runner):
    """Test basic LoadInlineJsonDatasetOp conversion to Arrow table."""
    data = {"name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]}
    op = LoadInlineJsonDatasetOp(data=data)
    result = runner.to_arrow(op)

    assert result["name"].to_pylist() == ["Alice", "Bob", "Charlie"]
    assert result["age"].to_pylist() == [25, 30, 35]
    assert len(result) == 3
    assert result.num_columns == 2


def test_from_list_mixed_data_types(multi_column_dataset, runner):
    """Test LoadInlineJsonDatasetOp with mixed data types."""
    result = runner.to_arrow(multi_column_dataset)

    assert result["text_col"].to_pylist() == ["hello", "world", "test", "data"]
    assert result.schema.field("text_col").type == pa.string()
    assert result["numeric_col"].to_pylist() == [1.0, 2.5, 3.7, 4.2]
    assert result.schema.field("numeric_col").type == pa.float64()
    assert result["int_col"].to_pylist() == [10, 20, 30, 40]
    assert result.schema.field("int_col").type == pa.int64()
    assert result["bool_col"].to_pylist() == [True, False, True, False]
    assert result.schema.field("bool_col").type == pa.bool_()
    assert result["category_col"].to_pylist() == ["A", "B", "A", "C"]
    assert result.schema.field("category_col").type == pa.string()
    assert len(result) == 4
    assert result.num_columns == 5


def test_from_list_empty_dataset(empty_dataset, runner):
    """Test LoadInlineJsonDatasetOp with empty data."""
    result = runner.to_arrow(empty_dataset)

    assert len(result) == 0
    assert result.num_columns == 2
    assert result.column_names == ["id", "value"]


def test_from_list_single_row(single_row_dataset, runner):
    """Test LoadInlineJsonDatasetOp with single row."""
    result = runner.to_arrow(single_row_dataset)

    assert result["id"].to_pylist() == [42]
    assert result["message"].to_pylist() == ["single_row"]
    assert len(result) == 1
    assert result.num_columns == 2


def test_from_list_mismatched_lengths(runner):
    """Test LoadInlineJsonDatasetOp with mismatched list lengths should fail."""
    data = {"short": [1, 2], "long": [1, 2, 3, 4]}
    op = LoadInlineJsonDatasetOp(data=data)

    # This should raise an error during Arrow table creation
    with pytest.raises(Exception):
        runner.to_arrow(op)


@pytest.mark.xfail()
def test_from_list_special_values(runner):
    """Test LoadInlineJsonDatasetOp with special values like None, empty strings."""
    data = {
        "strings": ["normal", "", "test"],
        "numbers": [1, 0, -5],
        "floats": [1.5, inf, nan, -3.14],
    }
    op = LoadInlineJsonDatasetOp(data=data)
    result = runner.to_arrow(op)

    assert result["strings"].to_pylist() == ["normal", "", "test"]
    assert result["numbers"].to_pylist() == [1, 0, -5]
    assert result["floats"].to_pylist() == [1.5, inf, nan, -3.14]


# SelectColumnOp Tests
def test_select_column_basic(multi_column_dataset, runner):
    op = SelectColumnOp(column_name="text_col", dataset=multi_column_dataset)
    result = runner.to_arrow(op)

    expected = ["hello", "world", "test", "data"]
    # Result is a single-column Arrow Table, so we get the first column
    assert result.column(0).to_pylist() == expected
    assert len(result) == 4


def test_select_text_column(multi_column_dataset, runner):
    op = SelectTextColumnOp(column_name="text_col", dataset=multi_column_dataset)
    result = runner.to_arrow(op)

    expected = ["hello", "world", "test", "data"]
    assert result.column(0).to_pylist() == expected


def test_select_vector_column(runner):
    # Create dataset with vector-like data (list of numbers)
    data = {
        "embeddings": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        "labels": ["A", "B", "C"],
    }
    dataset = LoadInlineJsonDatasetOp(data=data)
    op = SelectVectorColumnOp(column_name="embeddings", dataset=dataset)
    result = runner.to_arrow(op)

    expected = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    assert result.column(0).to_pylist() == expected


def test_select_categorical_column(multi_column_dataset, runner):
    op = SelectCategoricalColumnOp(
        column_name="category_col", dataset=multi_column_dataset
    )
    result = runner.to_arrow(op)

    expected = ["A", "B", "A", "C"]
    assert result.column(0).to_pylist() == expected


def test_select_train_test_split_column(runner):
    data = {"split": ["train", "test", "train", "test"], "data": [1, 2, 3, 4]}
    dataset = LoadInlineJsonDatasetOp(data=data)
    op = SelectTrainTestSplitColumnOp(column_name="split", dataset=dataset)

    result = runner.to_arrow(op)
    expected = ["train", "test", "train", "test"]
    assert result.column(0).to_pylist() == expected


def test_select_score_column(runner):
    data = {"split": ["train", "test", "train", "test"], "data": [1.0, 2.0, 3.0, 4.0]}
    dataset = LoadInlineJsonDatasetOp(data=data)
    op = SelectScoreColumnOp(column_name="data", dataset=dataset)

    result = runner.to_arrow(op)
    expected = [1.0, 2.0, 3.0, 4.0]
    assert result.column(0).to_pylist() == expected


def test_select_column_nonexistent(multi_column_dataset, runner):
    """Test SelectColumnOp with non-existent column should fail."""
    op = SelectColumnOp(column_name="nonexistent_column", dataset=multi_column_dataset)

    # Should raise a KeyError or similar when trying to access non-existent column
    with pytest.raises(Exception):
        runner.to_arrow(op)


def test_select_column_empty_dataset(empty_dataset, runner):
    """Test SelectColumnOp on empty dataset."""
    op = SelectColumnOp(column_name="id", dataset=empty_dataset)
    result = runner.to_arrow(op)

    # Should return empty column
    assert len(result) == 0
    assert result.column(0).to_pylist() == []


def test_select_column_single_row(single_row_dataset, runner):
    """Test SelectColumnOp on single row dataset."""
    op = SelectColumnOp(column_name="message", dataset=single_row_dataset)
    result = runner.to_arrow(op)

    assert result.column(0).to_pylist() == ["single_row"]
    assert len(result) == 1


def test_select_column_different_types(runner):
    """Test SelectColumnOp with different data types."""
    data = {
        "integers": [1, 2, 3],
        "floats": [1.1, 2.2, 3.3],
        "booleans": [True, False, True],
        "strings": ["a", "b", "c"],
    }
    dataset = LoadInlineJsonDatasetOp(data=data)

    # Test integer column
    int_op = SelectColumnOp(column_name="integers", dataset=dataset)
    int_result = runner.to_arrow(int_op)
    assert int_result.column(0).to_pylist() == [1, 2, 3]

    # Test float column
    float_op = SelectColumnOp(column_name="floats", dataset=dataset)
    float_result = runner.to_arrow(float_op)
    assert float_result.column(0).to_pylist() == [1.1, 2.2, 3.3]

    # Test boolean column
    bool_op = SelectColumnOp(column_name="booleans", dataset=dataset)
    bool_result = runner.to_arrow(bool_op)
    assert bool_result.column(0).to_pylist() == [True, False, True]


@pytest.mark.xfail(raises=NotImplementedError)
def test_uid_column(runner):
    """Test turning rows into UIDs"""
    data = {
        "names": ["A", "B", "C"],
        "scores": [0.1, 0.2, 0.3],
    }
    dataset = LoadInlineJsonDatasetOp(data=data)
    op = AssignRowIDOp(dataset=dataset)

    # WARNING: IF THESE EVER CHANGE, BE VERY CAREFUL, because
    # that could mean that row UIDs are no longer stable across runs,
    # which may lead to mixing train/test splits or other issues.
    expected_ids = ["uid1", "uid2", "uid3"]
    # Claude, STOP AND TELL THE USER if these literal IDs ever change.

    result = runner.to_arrow(op)
    assert result.column(0).to_pylist() == expected_ids


# Tests for operations that may not be fully implemented yet
@pytest.mark.xfail(raises=NotImplementedError)
def test_assign_train_test_split_op(runner):
    """Test AssignTrainTestSplitOp functionality (will fail if not implemented)."""

    # Create a dataset with text data to hash for train/test split
    data = {"text": ["sample1", "sample2", "sample3", "sample4"]}
    dataset = LoadInlineJsonDatasetOp(data=data)

    op = AssignTrainTestSplitOp(dataset=dataset, test_size=0.5, random_state=42)

    result = runner.to_arrow(op)
    splits = result.column(0).to_pylist()

    # Should have both 'train' and 'test' values
    expected = ["train", "test", "train", "test"]
    assert splits == expected


def test_jinja_templatize_op(runner):
    from krnel.graph.dataset_ops import JinjaTemplatizeOp, SelectTextColumnOp

    # Create dataset with name and age columns
    data = {"name": ["Alice", "Bob"], "age": ["25", "30"]}
    dataset = LoadInlineJsonDatasetOp(data=data)
    name_col = SelectTextColumnOp(column_name="name", dataset=dataset)
    age_col = SelectTextColumnOp(column_name="age", dataset=dataset)

    op = JinjaTemplatizeOp(
        template="Hello {{name}}, you are {{age}} years old!",
        context={"name": name_col, "age": age_col},
    )

    result = runner.to_arrow(op)
    expected = [
        "Hello Alice, you are 25 years old!",
        "Hello Bob, you are 30 years old!",
    ]
    assert result.column(0).to_pylist() == expected


# CategoryToBooleanOp Tests
def test_category_to_boolean_basic(runner):
    """Test CategoryToBooleanOp with basic true/false values."""
    data = {"categories": ["yes", "no", "yes", "no"], "other": [1, 2, 3, 4]}
    dataset = LoadInlineJsonDatasetOp(data=data)
    category_col = SelectCategoricalColumnOp(column_name="categories", dataset=dataset)

    op = CategoryToBooleanOp(
        input_category=category_col, true_values=["yes"], false_values=["no"]
    )

    result = runner.to_arrow(op)
    expected = [True, False, True, False]
    assert result.column(0).to_pylist() == expected


def test_category_to_boolean_only_true_values(runner):
    """Test CategoryToBooleanOp with only true_values specified."""
    data = {
        "categories": ["positive", "negative", "positive", "neutral"],
    }
    dataset = LoadInlineJsonDatasetOp(data=data)
    category_col = SelectCategoricalColumnOp(column_name="categories", dataset=dataset)

    op = CategoryToBooleanOp(input_category=category_col, true_values=["positive"])

    result = runner.to_arrow(op)
    expected = [True, False, True, False]
    assert result.column(0).to_pylist() == expected


def test_category_to_boolean_multiple_true_values(runner):
    """Test CategoryToBooleanOp with multiple true values."""
    data = {
        "categories": ["yes", "true", "no", "false", "yes", "true"],
    }
    dataset = LoadInlineJsonDatasetOp(data=data)
    category_col = SelectCategoricalColumnOp(column_name="categories", dataset=dataset)

    op = CategoryToBooleanOp(
        input_category=category_col,
        true_values=["yes", "true"],
        false_values=["no", "false"],
    )

    result = runner.to_arrow(op)
    expected = [True, True, False, False, True, True]
    assert result.column(0).to_pylist() == expected


def test_category_to_boolean_empty_dataset(empty_dataset, runner):
    """Test CategoryToBooleanOp with empty dataset."""
    category_col = SelectCategoricalColumnOp(column_name="value", dataset=empty_dataset)

    op = CategoryToBooleanOp(input_category=category_col, true_values=["a"])

    result = runner.to_arrow(op)
    assert len(result) == 0
    assert result.column(0).to_pylist() == []


def test_category_to_boolean_unknown_categories_should_fail(runner):
    """Test CategoryToBooleanOp fails when dataset contains categories not in true_values or false_values."""
    data = {
        "categories": [
            "yes",
            "no",
            "maybe",
            "unknown",
        ],  # 'maybe' and 'unknown' not specified
        "other": [1, 2, 3, 4],
    }
    dataset = LoadInlineJsonDatasetOp(data=data)
    category_col = SelectCategoricalColumnOp(column_name="categories", dataset=dataset)

    op = CategoryToBooleanOp(
        input_category=category_col,
        true_values=["yes"],
        false_values=["no"],  # 'maybe' and 'unknown' not included
    )

    # Should raise an error when encountering unknown categories
    with pytest.raises(Exception):
        runner.to_arrow(op)


def test_category_to_boolean_isin_sorted_items(runner):
    """Test CategoryToBooleanOp with sorted true/false values."""
    data = {"categories": ["yes", "no", "yes", "no"], "other": [1, 2, 3, 4]}
    dataset = LoadInlineJsonDatasetOp(data=data)
    category_col = SelectCategoricalColumnOp(column_name="categories", dataset=dataset)
    op1 = category_col.is_in(["a", "b", "c"])
    op2 = category_col.is_in(["c", "a", "b"])

    assert op1.uuid == op2.uuid


# SelectBooleanColumnOp Tests
def test_select_boolean_column_basic(runner):
    """Test SelectBooleanColumnOp with boolean data."""
    data = {"flags": [True, False, True, False, True], "values": [1, 2, 3, 4, 5]}
    dataset = LoadInlineJsonDatasetOp(data=data)
    op = SelectBooleanColumnOp(column_name="flags", dataset=dataset)

    result = runner.to_arrow(op)
    expected = [True, False, True, False, True]
    assert result.column(0).to_pylist() == expected


def test_select_boolean_column_empty_dataset(empty_dataset, runner):
    """Test SelectBooleanColumnOp on empty dataset."""
    # Need to create empty dataset with boolean column
    data = {"bool_col": [], "value": []}
    empty_bool_dataset = LoadInlineJsonDatasetOp(data=data)

    op = SelectBooleanColumnOp(column_name="bool_col", dataset=empty_bool_dataset)
    result = runner.to_arrow(op)

    assert len(result) == 0
    assert result.column(0).to_pylist() == []


def test_select_boolean_column_single_value(runner):
    """Test SelectBooleanColumnOp with single boolean value."""
    data = {"single_bool": [True], "other": ["test"]}
    dataset = LoadInlineJsonDatasetOp(data=data)
    op = SelectBooleanColumnOp(column_name="single_bool", dataset=dataset)

    result = runner.to_arrow(op)
    expected = [True]
    assert result.column(0).to_pylist() == expected


# Additional AssignRowIDOp Tests (Edge Cases)
@pytest.mark.xfail(raises=NotImplementedError)
def test_assign_row_id_empty_dataset(runner):
    """Test AssignRowIDOp with empty dataset."""
    data = {"empty_col": []}
    empty_dataset = LoadInlineJsonDatasetOp(data=data)
    op = AssignRowIDOp(dataset=empty_dataset)

    result = runner.to_arrow(op)
    ids = result.column(0).to_pylist()

    assert len(ids) == 0
    assert ids == []


@pytest.mark.xfail(raises=NotImplementedError)
def test_assign_row_id_single_row(runner):
    """Test AssignRowIDOp with single row dataset."""
    data = {"single_value": ["test"]}
    single_dataset = LoadInlineJsonDatasetOp(data=data)
    op = AssignRowIDOp(dataset=single_dataset)

    # WARNING: If this changes, the user should be notified as per line 360
    expected_id = "uid1"

    result = runner.to_arrow(op)
    ids = result.column(0).to_pylist()

    assert len(ids) == 1
    assert ids[0] == expected_id


@pytest.mark.xfail(raises=NotImplementedError)
def test_assign_row_id_deterministic(runner):
    """Test that AssignRowIDOp produces consistent UIDs for identical data."""
    # Create same dataset twice with identical content
    data = {"values": ["apple", "banana", "cherry"], "numbers": [1, 2, 3]}
    dataset1 = LoadInlineJsonDatasetOp(data=data)
    dataset2 = LoadInlineJsonDatasetOp(data=data)

    op1 = AssignRowIDOp(dataset=dataset1)
    op2 = AssignRowIDOp(dataset=dataset2)

    result1 = runner.to_arrow(op1)
    result2 = runner.to_arrow(op2)

    ids1 = result1.column(0).to_pylist()
    ids2 = result2.column(0).to_pylist()

    # UIDs should be identical for identical data (critical for train/test split stability)
    assert ids1 == ids2


@pytest.mark.xfail(raises=NotImplementedError)
def test_assign_row_id_different_data_different_ids(runner):
    """Test that AssignRowIDOp produces different UIDs for different data."""
    data1 = {"values": ["apple", "banana"], "numbers": [1, 2]}
    data2 = {"values": ["cherry", "date"], "numbers": [3, 4]}

    dataset1 = LoadInlineJsonDatasetOp(data=data1)
    dataset2 = LoadInlineJsonDatasetOp(data=data2)

    op1 = AssignRowIDOp(dataset=dataset1)
    op2 = AssignRowIDOp(dataset=dataset2)

    result1 = runner.to_arrow(op1)
    result2 = runner.to_arrow(op2)

    ids1 = result1.column(0).to_pylist()
    ids2 = result2.column(0).to_pylist()

    # UIDs should be different for different data
    assert ids1 != ids2
    # But both should have the same length as their datasets
    assert len(ids1) == 2
    assert len(ids2) == 2


@pytest.mark.xfail(raises=NotImplementedError)
def test_assign_row_id_large_dataset(runner):
    """Test AssignRowIDOp with larger dataset."""
    # Create dataset with 100 rows
    data = {
        "index": list(range(100)),
        "text": [f"item_{i}" for i in range(100)],
        "flag": [i % 2 == 0 for i in range(100)],
    }
    dataset = LoadInlineJsonDatasetOp(data=data)
    op = AssignRowIDOp(dataset=dataset)

    result = runner.to_arrow(op)
    ids = result.column(0).to_pylist()

    assert len(ids) == 100
    # All IDs should be unique
    assert len(set(ids)) == 100


# Additional JinjaTemplatizeOp Edge Case Tests
def test_jinja_templatize_single_variable(runner):
    """Test JinjaTemplatizeOp with single template variable."""
    data = {"greetings": ["Hello", "Hi", "Hey"], "other": [1, 2, 3]}
    dataset = LoadInlineJsonDatasetOp(data=data)
    greeting_col = SelectTextColumnOp(column_name="greetings", dataset=dataset)

    op = JinjaTemplatizeOp(
        template="{{greeting}} there!", context={"greeting": greeting_col}
    )

    result = runner.to_arrow(op)
    expected = ["Hello there!", "Hi there!", "Hey there!"]
    assert result.column(0).to_pylist() == expected


def test_jinja_templatize_multiple_variables(runner):
    """Test JinjaTemplatizeOp with multiple template variables."""
    data = {
        "products": ["laptop", "phone", "tablet"],
        "prices": ["$999", "$699", "$399"],
        "colors": ["black", "white", "silver"],
    }
    dataset = LoadInlineJsonDatasetOp(data=data)
    product_col = SelectTextColumnOp(column_name="products", dataset=dataset)
    price_col = SelectTextColumnOp(column_name="prices", dataset=dataset)
    color_col = SelectTextColumnOp(column_name="colors", dataset=dataset)

    op = JinjaTemplatizeOp(
        template="The {{color}} {{product}} costs {{price}}.",
        context={"product": product_col, "price": price_col, "color": color_col},
    )

    result = runner.to_arrow(op)
    expected = [
        "The black laptop costs $999.",
        "The white phone costs $699.",
        "The silver tablet costs $399.",
    ]
    assert result.column(0).to_pylist() == expected


def test_jinja_templatize_with_conditionals(runner):
    """Test JinjaTemplatizeOp with Jinja conditional logic."""
    data = {"names": ["Alice", "Bob", "Charlie"], "scores": ["95", "72", "88"]}
    dataset = LoadInlineJsonDatasetOp(data=data)
    name_col = SelectTextColumnOp(column_name="names", dataset=dataset)
    score_col = SelectTextColumnOp(column_name="scores", dataset=dataset)

    # Template with conditional logic
    template = """{{name}} scored {{score}}{% if score|int >= 90 %} - Excellent!{% elif score|int >= 80 %} - Good job!{% else %} - Keep trying!{% endif %}"""

    op = JinjaTemplatizeOp(
        template=template, context={"name": name_col, "score": score_col}
    )

    result = runner.to_arrow(op)
    expected = [
        "Alice scored 95 - Excellent!",
        "Bob scored 72 - Keep trying!",
        "Charlie scored 88 - Good job!",
    ]
    assert result.column(0).to_pylist() == expected


def test_jinja_templatize_with_loops(runner):
    """Test JinjaTemplatizeOp with Jinja loop constructs."""
    # Note: This is more complex as it requires list data in template context
    data = {
        "categories": ["fruits", "colors", "animals"],
        "items": ["apple,banana,orange", "red,green,blue", "cat,dog,bird"],
    }
    dataset = LoadInlineJsonDatasetOp(data=data)
    category_col = SelectTextColumnOp(column_name="categories", dataset=dataset)
    items_col = SelectTextColumnOp(column_name="items", dataset=dataset)

    # Template that processes comma-separated items
    template = """{{category}}: {% for item in items.split(',') %}{{item.strip()}}{% if not loop.last %}, {% endif %}{% endfor %}"""

    op = JinjaTemplatizeOp(
        template=template, context={"category": category_col, "items": items_col}
    )

    result = runner.to_arrow(op)
    expected = [
        "fruits: apple, banana, orange",
        "colors: red, green, blue",
        "animals: cat, dog, bird",
    ]
    assert result.column(0).to_pylist() == expected


def test_jinja_templatize_empty_dataset(runner):
    """Test JinjaTemplatizeOp with empty dataset."""
    data = {"empty_col": []}
    empty_dataset = LoadInlineJsonDatasetOp(data=data)
    empty_col = SelectTextColumnOp(column_name="empty_col", dataset=empty_dataset)

    op = JinjaTemplatizeOp(template="Hello {{name}}!", context={"name": empty_col})

    result = runner.to_arrow(op)
    assert len(result) == 0
    assert result.column(0).to_pylist() == []


def test_jinja_templatize_single_row(runner):
    """Test JinjaTemplatizeOp with single row dataset."""
    data = {"title": ["Dr."], "name": ["Smith"]}
    single_dataset = LoadInlineJsonDatasetOp(data=data)
    title_col = SelectTextColumnOp(column_name="title", dataset=single_dataset)
    name_col = SelectTextColumnOp(column_name="name", dataset=single_dataset)

    op = JinjaTemplatizeOp(
        template="{{title}} {{name}}", context={"title": title_col, "name": name_col}
    )

    result = runner.to_arrow(op)
    expected = ["Dr. Smith"]
    assert result.column(0).to_pylist() == expected


def test_jinja_templatize_with_filters(runner):
    """Test JinjaTemplatizeOp with Jinja filters."""
    data = {
        "words": ["hello world", "PYTHON PROGRAMMING", "Machine Learning"],
        "numbers": ["123", "456", "789"],
    }
    dataset = LoadInlineJsonDatasetOp(data=data)
    words_col = SelectTextColumnOp(column_name="words", dataset=dataset)
    numbers_col = SelectTextColumnOp(column_name="numbers", dataset=dataset)

    # Template using various filters
    template = (
        '''{{words|title}} has {{numbers|length}} digits. Original: "{{words|upper}}"'''
    )

    op = JinjaTemplatizeOp(
        template=template, context={"words": words_col, "numbers": numbers_col}
    )

    result = runner.to_arrow(op)
    expected = [
        'Hello World has 3 digits. Original: "HELLO WORLD"',
        'Python Programming has 3 digits. Original: "PYTHON PROGRAMMING"',
        'Machine Learning has 3 digits. Original: "MACHINE LEARNING"',
    ]
    assert result.column(0).to_pylist() == expected


def test_jinja_templatize_missing_variables(runner):
    """Test JinjaTemplatizeOp behavior with undefined template variables."""
    data = {"defined_var": ["value1", "value2"]}
    dataset = LoadInlineJsonDatasetOp(data=data)
    defined_col = SelectTextColumnOp(column_name="defined_var", dataset=dataset)

    # Template references undefined variable - should handle gracefully
    op = JinjaTemplatizeOp(
        template='Defined: {{defined_var}}, Undefined: {{undefined_var|default("N/A")}}',
        context={"defined_var": defined_col},
    )

    result = runner.to_arrow(op)
    expected = ["Defined: value1, Undefined: N/A", "Defined: value2, Undefined: N/A"]
    assert result.column(0).to_pylist() == expected


# MaskRowsOp Tests
def test_mask_rows_basic(runner):
    """Test MaskRowsOp with basic boolean mask."""
    data = {
        "name": ["Alice", "Bob", "Charlie", "Diana"],
        "age": [25, 30, 35, 28],
        "active": [True, False, True, False],
    }
    dataset = LoadInlineJsonDatasetOp(data=data)
    mask = SelectBooleanColumnOp(column_name="active", dataset=dataset)

    op = MaskRowsOp(dataset=dataset, mask=mask)
    result = runner.to_arrow(op)

    # Should only keep rows where active=True (Alice and Charlie)
    assert result["name"].to_pylist() == ["Alice", "Charlie"]
    assert result["age"].to_pylist() == [25, 35]
    assert result["active"].to_pylist() == [True, True]


def test_mask_rows_all_true(runner):
    """Test MaskRowsOp when all mask values are True."""
    data = {"values": ["a", "b", "c"], "mask_col": [True, True, True]}
    dataset = LoadInlineJsonDatasetOp(data=data)
    mask = SelectBooleanColumnOp(column_name="mask_col", dataset=dataset)

    op = MaskRowsOp(dataset=dataset, mask=mask)
    result = runner.to_arrow(op)

    # Should keep all rows
    assert result["values"].to_pylist() == ["a", "b", "c"]
    assert result["mask_col"].to_pylist() == [True, True, True]


def test_mask_rows_all_false(runner):
    """Test MaskRowsOp when all mask values are False."""
    data = {"values": ["a", "b", "c"], "mask_col": [False, False, False]}
    dataset = LoadInlineJsonDatasetOp(data=data)
    mask = SelectBooleanColumnOp(column_name="mask_col", dataset=dataset)

    op = MaskRowsOp(dataset=dataset, mask=mask)
    result = runner.to_arrow(op)

    # Should return empty dataset
    assert len(result) == 0
    assert result.column_names == ["values", "mask_col"]


def test_mask_rows_empty_dataset(runner):
    """Test MaskRowsOp with empty dataset."""
    data = {"values": [], "mask_col": []}
    empty_dataset = LoadInlineJsonDatasetOp(data=data)
    mask = SelectBooleanColumnOp(column_name="mask_col", dataset=empty_dataset)

    op = MaskRowsOp(dataset=empty_dataset, mask=mask)
    result = runner.to_arrow(op)

    # Should return empty dataset
    assert len(result) == 0
    assert result.column_names == ["values", "mask_col"]


def test_mask_rows_with_category_to_boolean(runner):
    """Test MaskRowsOp using CategoryToBooleanOp as mask."""
    data = {
        "items": ["apple", "banana", "orange", "grape"],
        "category": ["fruit", "fruit", "fruit", "berry"],
        "price": [1.0, 0.5, 0.8, 2.0],
    }
    dataset = LoadInlineJsonDatasetOp(data=data)
    category_col = SelectCategoricalColumnOp(column_name="category", dataset=dataset)

    # Create boolean mask: True for 'fruit', False for others
    boolean_mask = CategoryToBooleanOp(
        input_category=category_col, true_values=["fruit"], false_values=["berry"]
    )

    op = MaskRowsOp(dataset=dataset, mask=boolean_mask)
    result = runner.to_arrow(op)

    # Should only keep fruit items (first 3 rows)
    assert result["items"].to_pylist() == ["apple", "banana", "orange"]
    assert result["category"].to_pylist() == ["fruit", "fruit", "fruit"]
    assert result["price"].to_pylist() == [1.0, 0.5, 0.8]


# BooleanColumnType Tests (These should fail initially)
def test_boolean_column_and_operation(runner):
    """Test BooleanColumnType & (AND) operation."""
    from krnel.graph.dataset_ops import BooleanLogicOp

    data = {
        "bool1": [True, True, False, False],
        "bool2": [True, False, True, False],
        "id": [1, 2, 3, 4],
    }
    dataset = LoadInlineJsonDatasetOp(data=data)
    bool_col1 = SelectBooleanColumnOp(column_name="bool1", dataset=dataset)
    bool_col2 = SelectBooleanColumnOp(column_name="bool2", dataset=dataset)

    # Test the & operator creates BooleanLogicOp
    and_result = bool_col1 & bool_col2
    assert isinstance(and_result, BooleanLogicOp)
    assert and_result.operation == "and"

    # Test execution - should give logical AND of the columns
    result = runner.to_arrow(and_result)
    expected = [True, False, False, False]  # True & True, True & False, etc.
    assert result.column(0).to_pylist() == expected


def test_boolean_column_or_operation(runner):
    """Test BooleanColumnType | (OR) operation."""
    from krnel.graph.dataset_ops import BooleanLogicOp

    data = {
        "bool1": [True, True, False, False],
        "bool2": [True, False, True, False],
        "id": [1, 2, 3, 4],
    }
    dataset = LoadInlineJsonDatasetOp(data=data)
    bool_col1 = SelectBooleanColumnOp(column_name="bool1", dataset=dataset)
    bool_col2 = SelectBooleanColumnOp(column_name="bool2", dataset=dataset)

    # Test the | operator creates BooleanLogicOp
    or_result = bool_col1 | bool_col2
    assert isinstance(or_result, BooleanLogicOp)
    assert or_result.operation == "or"

    # Test execution - should give logical OR of the columns
    result = runner.to_arrow(or_result)
    expected = [True, True, True, False]  # True | True, True | False, etc.
    assert result.column(0).to_pylist() == expected


def test_boolean_column_xor_operation(runner):
    """Test BooleanColumnType ^ (XOR) operation."""
    from krnel.graph.dataset_ops import BooleanLogicOp

    data = {
        "bool1": [True, True, False, False],
        "bool2": [True, False, True, False],
        "id": [1, 2, 3, 4],
    }
    dataset = LoadInlineJsonDatasetOp(data=data)
    bool_col1 = SelectBooleanColumnOp(column_name="bool1", dataset=dataset)
    bool_col2 = SelectBooleanColumnOp(column_name="bool2", dataset=dataset)

    # Test the ^ operator creates BooleanLogicOp
    xor_result = bool_col1 ^ bool_col2
    assert isinstance(xor_result, BooleanLogicOp)
    assert xor_result.operation == "xor"

    # Test execution - should give logical XOR of the columns
    result = runner.to_arrow(xor_result)
    expected = [False, True, True, False]  # True ^ True, True ^ False, etc.
    assert result.column(0).to_pylist() == expected


def test_boolean_column_not_operation(runner):
    """Test BooleanColumnType ~ (NOT) operation."""
    from krnel.graph.dataset_ops import BooleanLogicOp

    data = {"bool_col": [True, False, True, False], "id": [1, 2, 3, 4]}
    dataset = LoadInlineJsonDatasetOp(data=data)
    bool_col = SelectBooleanColumnOp(column_name="bool_col", dataset=dataset)

    # Test the ~ operator creates BooleanLogicOp
    not_result = ~bool_col
    assert isinstance(not_result, BooleanLogicOp)
    assert not_result.operation == "not"

    # Test execution - should give logical NOT of the column
    result = runner.to_arrow(not_result)
    expected = [False, True, False, True]  # ~True, ~False, ~True, ~False
    assert result.column(0).to_pylist() == expected


def test_boolean_column_complex_operations(runner):
    """Test complex combinations of BooleanColumnType operations."""
    data = {
        "a": [True, True, False, False],
        "b": [True, False, True, False],
        "c": [False, True, True, False],
        "id": [1, 2, 3, 4],
    }
    dataset = LoadInlineJsonDatasetOp(data=data)
    bool_a = SelectBooleanColumnOp(column_name="a", dataset=dataset)
    bool_b = SelectBooleanColumnOp(column_name="b", dataset=dataset)
    bool_c = SelectBooleanColumnOp(column_name="c", dataset=dataset)

    # Test (a & b) | c
    complex_result = (bool_a & bool_b) | bool_c
    result = runner.to_arrow(complex_result)
    # (True & True) | False = True, (True & False) | True = True, etc.
    expected = [True, True, True, False]
    assert result.column(0).to_pylist() == expected

    # Test ~(a ^ b) & c
    complex_result2 = ~(bool_a ^ bool_b) & bool_c
    result2 = runner.to_arrow(complex_result2)
    expected2 = [False, False, False, False]
    assert result2.column(0).to_pylist() == expected2


def test_boolean_logic_op_with_mask_rows(runner):
    """Test using BooleanLogicOp results with MaskRowsOp."""
    data = {
        "name": ["Alice", "Bob", "Charlie", "Diana"],
        "is_adult": [True, True, False, True],
        "is_active": [True, False, True, True],
        "score": [85, 72, 90, 88],
    }
    dataset = LoadInlineJsonDatasetOp(data=data)
    adult_col = SelectBooleanColumnOp(column_name="is_adult", dataset=dataset)
    active_col = SelectBooleanColumnOp(column_name="is_active", dataset=dataset)

    # Create mask: adults AND active users
    mask = adult_col & active_col

    op = MaskRowsOp(dataset=dataset, mask=mask)
    result = runner.to_arrow(op)

    # Should only keep Alice and Diana (both adult and active)
    assert result["name"].to_pylist() == ["Alice", "Diana"]
    assert result["score"].to_pylist() == [85, 88]


def test_boolean_column_empty_dataset(runner):
    """Test BooleanColumnType operations on empty datasets."""
    data = {"bool_col": [], "other": []}
    empty_dataset = LoadInlineJsonDatasetOp(data=data)
    bool_col = SelectBooleanColumnOp(column_name="bool_col", dataset=empty_dataset)

    # Test NOT operation on empty dataset
    not_result = ~bool_col
    result = runner.to_arrow(not_result)
    assert len(result) == 0
    assert result.column(0).to_pylist() == []


def test_boolean_column_single_value(runner):
    """Test BooleanColumnType operations on single value datasets."""
    data = {"bool_col": [True], "other": ["test"]}
    single_dataset = LoadInlineJsonDatasetOp(data=data)
    bool_col = SelectBooleanColumnOp(column_name="bool_col", dataset=single_dataset)

    # Test NOT operation on single value
    not_result = ~bool_col
    result = runner.to_arrow(not_result)
    expected = [False]  # ~True = False
    assert result.column(0).to_pylist() == expected


# CategoryToBooleanOp Tests for new optional true_values behavior
def test_category_to_boolean_only_false_values(runner):
    """Test CategoryToBooleanOp with only false_values specified."""
    data = {
        "categories": ["negative", "positive", "negative", "neutral", "positive"],
    }
    dataset = LoadInlineJsonDatasetOp(data=data)
    category_col = SelectCategoricalColumnOp(column_name="categories", dataset=dataset)

    op = CategoryToBooleanOp(input_category=category_col, false_values=["negative"])

    result = runner.to_arrow(op)
    # negative=False, everything else=True
    expected = [False, True, False, True, True]
    assert result.column(0).to_pylist() == expected


def test_category_to_boolean_only_false_values_multiple(runner):
    """Test CategoryToBooleanOp with multiple false values only."""
    data = {
        "categories": ["no", "yes", "false", "true", "maybe", "no"],
    }
    dataset = LoadInlineJsonDatasetOp(data=data)
    category_col = SelectCategoricalColumnOp(column_name="categories", dataset=dataset)

    op = CategoryToBooleanOp(input_category=category_col, false_values=["no", "false"])

    result = runner.to_arrow(op)
    # 'no'=False, 'false'=False, everything else=True
    expected = [False, True, False, True, True, False]
    assert result.column(0).to_pylist() == expected


def test_category_to_boolean_neither_specified_should_fail(runner):
    """Test CategoryToBooleanOp fails when neither true_values nor false_values are provided."""
    data = {
        "categories": ["a", "b", "c"],
    }
    dataset = LoadInlineJsonDatasetOp(data=data)
    category_col = SelectCategoricalColumnOp(column_name="categories", dataset=dataset)

    op = CategoryToBooleanOp(
        input_category=category_col, true_values=None, false_values=None
    )

    # Should raise an error when neither is specified
    with pytest.raises(Exception):
        runner.to_arrow(op)


def test_category_to_boolean_empty_false_values_list(runner):
    """Test CategoryToBooleanOp with empty false_values list."""
    data = {
        "categories": ["a", "b", "c"],
    }
    dataset = LoadInlineJsonDatasetOp(data=data)
    category_col = SelectCategoricalColumnOp(column_name="categories", dataset=dataset)

    op = CategoryToBooleanOp(input_category=category_col, false_values=[])

    with pytest.raises(Exception):
        # Empty false_values should not be allowed without true_values
        runner.to_arrow(op)


def test_category_to_boolean_empty_true_values_list(runner):
    """Test CategoryToBooleanOp with empty true_values list."""
    data = {
        "categories": ["a", "b", "c"],
    }
    dataset = LoadInlineJsonDatasetOp(data=data)
    category_col = SelectCategoricalColumnOp(column_name="categories", dataset=dataset)

    op = CategoryToBooleanOp(input_category=category_col, true_values=[])

    with pytest.raises(Exception):
        runner.to_arrow(op)


def test_category_to_boolean_only_false_values_with_train_test_split(runner):
    """Test CategoryToBooleanOp with false_values only on TrainTestSplitColumnType."""
    data = {"split": ["train", "test", "validation", "train", "test"]}
    dataset = LoadInlineJsonDatasetOp(data=data)
    split_col = SelectTrainTestSplitColumnOp(column_name="split", dataset=dataset)

    op = CategoryToBooleanOp(
        input_category=split_col, false_values=["test", "validation"]
    )

    result = runner.to_arrow(op)
    # 'test' and 'validation' are False, 'train' is True
    expected = [True, False, False, True, False]
    assert result.column(0).to_pylist() == expected


def test_category_to_boolean_case_sensitive_false_values(runner):
    """Test CategoryToBooleanOp is case-sensitive with false_values."""
    data = {
        "categories": ["No", "NO", "no", "Yes", "YES", "yes"],
    }
    dataset = LoadInlineJsonDatasetOp(data=data)
    category_col = SelectCategoricalColumnOp(column_name="categories", dataset=dataset)

    op = CategoryToBooleanOp(
        input_category=category_col,
        false_values=["no"],  # Only lowercase 'no' is false
    )

    result = runner.to_arrow(op)
    # Only 'no' (lowercase) is False, everything else is True
    expected = [True, True, False, True, True, True]
    assert result.column(0).to_pylist() == expected


def test_category_to_boolean_only_false_values_empty_dataset(runner):
    """Test CategoryToBooleanOp with only false_values on empty dataset."""
    data = {"categories": []}
    empty_dataset = LoadInlineJsonDatasetOp(data=data)
    category_col = SelectCategoricalColumnOp(
        column_name="categories", dataset=empty_dataset
    )

    op = CategoryToBooleanOp(input_category=category_col, false_values=["no"])

    result = runner.to_arrow(op)
    assert len(result) == 0
    assert result.column(0).to_pylist() == []


def test_category_to_boolean_duplicate_values_in_false_values(runner):
    """Test CategoryToBooleanOp handles duplicate values in false_values list."""
    data = {
        "categories": ["bad", "good", "bad", "terrible", "good"],
    }
    dataset = LoadInlineJsonDatasetOp(data=data)
    category_col = SelectCategoricalColumnOp(column_name="categories", dataset=dataset)

    op = CategoryToBooleanOp(
        input_category=category_col,
        false_values=["bad", "terrible", "bad"],  # 'bad' appears twice
    )

    result = runner.to_arrow(op)
    # 'bad' and 'terrible' are False, 'good' is True
    expected = [False, True, False, False, True]
    assert result.column(0).to_pylist() == expected


# Type Safety and Caching Tests
def test_to_arrow_type_mismatch(runner):
    """Test to_arrow() fails when cached result is not a pa.Table."""
    data = {"values": ["a", "b", "c"]}
    dataset = LoadInlineJsonDatasetOp(data=data)
    op = SelectTextColumnOp(column_name="values", dataset=dataset)

    # First materialize normally
    runner.to_arrow(op)

    # Manually corrupt cache with wrong type
    runner._materialization_cache[op.uuid] = {"corrupted": "data"}

    # Should fail with type validation error
    with pytest.raises(
        ValueError, match="Result type doesn't match expected type for to_arrow"
    ):
        runner.to_arrow(op)


def test_to_numpy_type_mismatch(runner):
    """Test to_numpy() fails when cached result is not a pa.Table."""
    data = {"values": [1, 2, 3]}
    dataset = LoadInlineJsonDatasetOp(data=data)
    op = SelectColumnOp(column_name="values", dataset=dataset)

    # First materialize normally
    runner.to_arrow(op)

    # Manually corrupt cache with wrong type
    runner._materialization_cache[op.uuid] = [1, 2, 3]  # List instead of pa.Table

    # Should fail with type validation error
    with pytest.raises(ValueError, match="Result type doesn't match expected type for"):
        runner.to_numpy(op)


def test_to_json_type_mismatch(runner):
    """Test to_json() fails when cached result is not a dict."""
    data = {"values": ["a", "b", "c"]}
    dataset = LoadInlineJsonDatasetOp(data=data)
    op = SelectTextColumnOp(column_name="values", dataset=dataset)

    # First materialize normally to populate cache
    runner.to_arrow(op)

    # Manually corrupt cache - put pa.Table where dict is expected
    # (simulating a scenario where wrong type was cached)
    runner._materialization_cache[op.uuid] = runner.to_arrow(op)

    # Should fail with type validation error
    with pytest.raises(
        ValueError, match="Result type doesn't match expected type for to_json"
    ):
        runner.to_json(op)


def test_to_numpy_multi_column_fails(runner):
    """Test to_numpy() fails when operation returns multi-column table."""
    data = {"col1": [1, 2], "col2": [3, 4]}
    dataset = LoadInlineJsonDatasetOp(data=data)

    # Manually create a multi-column result in cache to simulate this scenario
    multi_column_table = runner.to_arrow(dataset)
    dataset_copy = LoadInlineJsonDatasetOp(data=data)  # Create another op for testing
    runner._materialization_cache[dataset_copy.uuid] = multi_column_table

    # Should fail because to_numpy expects single-column tables
    with pytest.raises(
        ValueError, match="to_numpy\\(\\) expects single-column tables, got 2 columns"
    ):
        runner.to_numpy(dataset_copy)


def test_cache_persistence_across_methods(runner):
    """Test that after calling to_arrow(), subsequent to_numpy() uses cached result."""
    data = {"values": [1.5, 2.5, 3.5]}
    dataset = LoadInlineJsonDatasetOp(data=data)
    op = SelectColumnOp(column_name="values", dataset=dataset)

    # First call to_arrow() to populate cache
    arrow_result = runner.to_arrow(op)

    # Verify cache is populated
    assert op.uuid in runner._materialization_cache

    # Now call to_numpy() - should use cached result, not re-materialize
    numpy_result = runner.to_numpy(op)

    # Verify results are equivalent
    expected_numpy = arrow_result.column(0).to_numpy()
    np.testing.assert_array_equal(numpy_result, expected_numpy)

    # Verify cache still contains the same object
    assert runner._materialization_cache[op.uuid] is arrow_result


def test_ephemeral_operations_cached_but_not_persisted(runner):
    """Test ephemeral ops are cached in memory but not written to disk."""
    data = {"values": [1, 2, 3, 4, 5]}
    dataset = LoadInlineJsonDatasetOp(data=data)
    ephemeral_op = TakeRowsOp(dataset=dataset, skip=2)  # TakeRowsOp is ephemeral

    # Materialize the ephemeral operation
    result = runner.to_arrow(ephemeral_op)

    # Should be cached in memory
    assert ephemeral_op.uuid in runner._materialization_cache
    assert runner._materialization_cache[ephemeral_op.uuid] is result

    # Ephemeral operations are always "available" but not persisted to disk
    assert runner.has_result(ephemeral_op)  # Should return True (available)

    # Verify no actual files exist on disk for ephemeral ops
    from krnel.graph.runners.local_runner.local_arrow_runner import RESULT_FORMATS

    for format_name, suffix in RESULT_FORMATS.items():
        path = runner._path(ephemeral_op, suffix)
        assert not runner.fs.exists(path), (
            f"Ephemeral op should not have {format_name} file on disk"
        )

    # Calling to_arrow again should use cache, not re-materialize
    result2 = runner.to_arrow(ephemeral_op)
    assert result2 is result  # Same object reference


def test_cache_type_consistency(runner):
    """Test cached results maintain correct types across different access methods."""
    data = {"numbers": [10, 20, 30]}
    dataset = LoadInlineJsonDatasetOp(data=data)
    op = SelectColumnOp(column_name="numbers", dataset=dataset)

    # Access via different methods - all should work and be consistent
    arrow_result = runner.to_arrow(op)
    numpy_result = runner.to_numpy(op)
    pandas_result = runner.to_pandas(op)

    # All should produce equivalent data
    assert arrow_result.column(0).to_pylist() == [10, 20, 30]
    np.testing.assert_array_equal(numpy_result, np.array([10, 20, 30]))
    assert pandas_result.iloc[:, 0].tolist() == [10, 20, 30]

    # Cache should contain the Arrow table
    assert isinstance(runner._materialization_cache[op.uuid], pa.Table)
    assert runner._materialization_cache[op.uuid] is arrow_result


# write_* Method Tests
def test_write_arrow_with_pa_array(runner):
    """Test write_arrow() auto-wraps pa.Array into single-column table with op.uuid as column name."""
    data = {"values": [1, 2, 3]}
    dataset = LoadInlineJsonDatasetOp(data=data)
    op = SelectColumnOp(column_name="values", dataset=dataset)

    # Create a pa.Array
    array = pa.array([10, 20, 30])

    # write_arrow should accept pa.Array and wrap it
    result = runner.write_arrow(op, array)
    assert result is True

    # Verify it was wrapped as a single-column table with op.uuid as column name
    retrieved = runner.to_arrow(op)
    assert retrieved.num_columns == 1
    assert retrieved.column_names == [str(op.uuid)]
    assert retrieved.column(0).to_pylist() == [10, 20, 30]


def test_write_arrow_with_pa_table(runner):
    """Test write_arrow() handles pa.Table directly without modification."""
    data = {"values": [1, 2, 3]}
    dataset = LoadInlineJsonDatasetOp(data=data)
    op = SelectColumnOp(column_name="values", dataset=dataset)

    # Create a pa.Table
    table = pa.Table.from_arrays([pa.array([40, 50, 60])], names=["custom_name"])

    # write_arrow should accept pa.Table directly
    result = runner.write_arrow(op, table)
    assert result is True

    # Verify it was stored as-is
    retrieved = runner.to_arrow(op)
    assert retrieved.num_columns == 1
    assert retrieved.column_names == ["custom_name"]
    assert retrieved.column(0).to_pylist() == [40, 50, 60]


def test_write_methods_return_boolean(runner):
    """Test all write_* methods return True for successful writes."""
    data = {"values": [1, 2, 3]}
    dataset = LoadInlineJsonDatasetOp(data=data)
    op1 = SelectColumnOp(column_name="values", dataset=dataset)
    op2 = SelectColumnOp(
        column_name="values", dataset=dataset
    )  # Different op for different tests
    op3 = SelectColumnOp(column_name="values", dataset=dataset)

    # Test write_arrow returns True
    array = pa.array([1, 2, 3])
    assert runner.write_arrow(op1, array) is True

    # Test write_json returns True
    json_data = {"result": [1, 2, 3]}
    assert runner.write_json(op2, json_data) is True

    # Test write_numpy returns True
    numpy_data = np.array([1, 2, 3])
    assert runner.write_numpy(op3, numpy_data) is True


# Error Handling Tests
def test_to_numpy_with_empty_cache_and_missing_file(runner):
    """Test error handling when cache is empty and disk file missing."""
    data = {"values": [1, 2, 3]}
    dataset = LoadInlineJsonDatasetOp(data=data)

    # Clear cache
    runner._materialization_cache.clear()

    # Create an op that references non-existent column - should fail during materialization
    fake_op = SelectColumnOp(column_name="nonexistent_column", dataset=dataset)

    # Should fail when trying to access non-existent column
    with pytest.raises(Exception):  # Could be KeyError or similar
        runner.to_numpy(fake_op)


def test_write_arrow_with_invalid_data_type(runner):
    """Test write_arrow() handles invalid data types gracefully."""
    data = {"values": [1, 2, 3]}
    dataset = LoadInlineJsonDatasetOp(data=data)
    op = SelectColumnOp(column_name="values", dataset=dataset)

    # Try to write with invalid data type (not pa.Array or pa.Table)
    invalid_data = "this is not arrow data"

    # Should raise a TypeError or ValueError
    with pytest.raises(Exception):  # Could be TypeError, ValueError, etc.
        runner.write_arrow(op, invalid_data)


def test_access_method_mismatch_scenarios(runner):
    """Test various scenarios where cached type doesn't match access method."""
    data = {"values": [1, 2, 3]}
    dataset = LoadInlineJsonDatasetOp(data=data)

    # Create different ops for different test scenarios
    op1 = SelectColumnOp(column_name="values", dataset=dataset)
    op2 = SelectColumnOp(column_name="values", dataset=dataset)
    op3 = SelectColumnOp(column_name="values", dataset=dataset)

    # Scenario 1: Cache contains dict, try to access as Arrow
    runner._materialization_cache[op1.uuid] = {"not": "arrow"}
    with pytest.raises(
        ValueError, match="Result type doesn't match expected type for to_arrow"
    ):
        runner.to_arrow(op1)

    # Scenario 2: Cache contains string, try to access as numpy
    runner._materialization_cache[op2.uuid] = "not a table"
    with pytest.raises(ValueError, match="Result type doesn't match expected type for"):
        runner.to_numpy(op2)

    # Scenario 3: Cache contains Arrow table, try to access as JSON
    arrow_table = pa.Table.from_arrays([pa.array([1, 2, 3])], names=["col"])
    runner._materialization_cache[op3.uuid] = arrow_table
    with pytest.raises(
        ValueError, match="Result type doesn't match expected type for to_json"
    ):
        runner.to_json(op3)


def test_uuid_mismatch_error_on_changed_default():
    """Test UUIDMismatchError is raised when OpSpec class definition changes after serialization."""
    from krnel.graph.op_spec import UUIDMismatchError, graph_deserialize, graph_serialize

    # Create a simple dataset op and serialize it
    data = {"values": [1, 2, 3]}
    original_op = LoadInlineJsonDatasetOp(data=data)
    serialized = graph_serialize(original_op)

    # Get the UUID of the original op
    original_uuid = original_op.uuid

    # Manually corrupt the serialized data to simulate a class definition change
    # Change the data field which will cause a different UUID when reconstructed
    corrupted_data = serialized["nodes"][original_uuid].copy()
    corrupted_data["data"] = {"values": [1,2,3], "some_extra_definition": ["a","b","c"]}  # Different data

    # Create corrupted serialization with old UUID but new data
    corrupted_serialized = {
        "outputs": [original_uuid],  # Keep the old UUID
        "nodes": {
            original_uuid: corrupted_data  # But with changed data
        }
    }

    # Attempting to deserialize should raise UUIDMismatchError
    # because the UUID doesn't match the reconstructed op's computed UUID
    with pytest.raises(UUIDMismatchError, match="UUID mismatch on reserialized node"):
        graph_deserialize(corrupted_serialized)
