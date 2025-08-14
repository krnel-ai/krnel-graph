# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

from typing import Any, TypeVar, Generic
from krnel.graph import OpSpec, EphemeralOpMixin
from krnel.graph.types import *

"""
List of operations related to datasets.

In `dataset_ops.py`, operations are defined as subclasses of `OpSpec`. Operations
that return objects of a specific type should inherit from that type, e.g.:

.. code-block:: python
    class TrainClassifierOp(ClassifierType):
        train_input: InputType

    class ApplyClassifierOp(ScoreColumnType):
        "produce output scores from a classifier"
        classifier: ClassifierType
        test_input: InputType

The types themselves and the API they follow are defined in `types.py`:

..  code-block:: python
    # in types.py:
    class SomeInputType(OpSpec):
        ...
        def train_classifier(self) -> ClassifierType:
            return TrainClassifierOp(some_input=some_input)

    class ClassifierType(OpSpec):
        ...
        def apply(self, input: InputType) -> ScoreColumnType:
            return ApplyClassifierOp(classifier=self, test_input=input)

"""


class LoadDatasetOp(DatasetType):
    """
    An operation that loads some specific, immutable dataset.

    Attributes:
        content_hash: A unique hash identifying the dataset's content.
    """
    content_hash: str


class SelectColumnOp(OpSpec, EphemeralOpMixin):
    """
    A single column from the input dataset.
    """
    column_name: str
    dataset: DatasetType

class SelectVectorColumnOp(SelectColumnOp, VectorColumnType):
    ...
class SelectTextColumnOp(SelectColumnOp, TextColumnType):
    ...
class SelectConversationColumnOp(SelectColumnOp, ConversationColumnType):
    ...
class SelectCategoricalColumnOp(SelectColumnOp, CategoricalColumnType):
    ...
class SelectTrainTestSplitColumnOp(SelectColumnOp, TrainTestSplitColumnType):
    ...
class SelectScoreColumnOp(SelectColumnOp, ScoreColumnType):
    ...
class SelectBooleanColumnOp(SelectColumnOp, BooleanColumnType):
    ...

class AssignRowIDOp(RowIDColumnType):
    """
    An operation that assigns a unique row ID to each row in the dataset.
    """
    dataset: DatasetType

class AssignTrainTestSplitOp(TrainTestSplitColumnType):
    """
    An operation that assigns a train/test split to a dataset column.

    To load the train/test split from a column in the database, use
    SelectTrainTestSplitColumnOp instead.
    """
    dataset: DatasetType
    test_size: float | int | None = None
    train_size: float | int | None = None
    random_state: int

class JinjaTemplatizeOp(TextColumnType):
    """
    An operation that templatizes a Jinja template with the given context.
    """
    template: str
    context: dict[str, TextColumnType]

class TakeRowsOp(DatasetType, EphemeralOpMixin):
    """
    Subsample the dataset by `skip` and `offset`, then take `num_rows` rows.
    """
    dataset: DatasetType
    skip: int = 1
    offset: int = 0
    num_rows: int | None = None

class MaskRowsOp(DatasetType, EphemeralOpMixin):
    """
    Filter rows in the dataset based on a boolean mask.

    The mask is a boolean column that indicates which rows to keep.
    """
    dataset: DatasetType
    mask: BooleanColumnType

class FromListOp(DatasetType):
    """
    An operation that creates a dataset from Python lists/dicts.
    Useful for testing and creating small datasets programmatically.
    """
    data: dict[str, list[Any]]

class CategoryToBooleanOp(BooleanColumnType):
    """
    An operation that converts a categorical column to a boolean column.

    This is useful for binary classification tasks where the categorical
    values represent two distinct classes.

    When both `true_values` and `false_values` are provided,
    the set of actual values must be a subset
    of `true_values.union(false_values)`.
    """
    input_category: CategoricalColumnType | TrainTestSplitColumnType
    true_values: set[str]
    false_values: set[str] | None = None