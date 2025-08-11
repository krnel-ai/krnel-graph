# Copyright (c) 2025 Krnel
# Points of Contact: 
#   - kimmy@krnel.ai

from typing import Any, TypeVar, Generic
from krnel.graph import OpSpec
from krnel.graph.types import *


# Graph spec
class LoadDatasetOp(DatasetType):
    """
    An operation that loads some specific, immutable dataset.
    """
    ...


class SelectColumnOp(OpSpec):
    """
    A single column from the input dataset.
    """
    column_name: str
    dataset: DatasetType

class SelectTextColumnOp(SelectColumnOp, TextColumnType):
    ...
class SelectTrainTestSplitColumnOp(SelectColumnOp, TrainTestSplitColumnType):
    ...
class SelectEmbeddingColumnOp(SelectColumnOp, VectorColumnType):
    ...
class SelectCategoricalColumnOp(SelectColumnOp, CategoricalColumnType):
    ...

class AssignTrainTestSplitOp(TrainTestSplitColumnType):
    """
    An operation that assigns a train/test split to a dataset column.

    To load the train/test split from a column in the database, use
    SelectTrainTestSplitColumnOp instead.
    """
    hash_column: TextColumnType
    test_size: float | int | None = None
    train_size: float | int | None = None
    random_state: int

class JinjaTemplatizeOp(TextColumnType):
    """
    An operation that templatizes a Jinja template with the given context.
    This can be used to create prompts, for example.
    """
    template: str
    context: dict[str, TextColumnType]

class TakeRowsOp(DatasetType):
    """
    Subsample the dataset by `skip`, then take `num_rows` rows.
    """
    dataset: DatasetType
    skip: int = 1
    num_rows: int | None = None

class FromListOp(DatasetType):
    """
    An operation that creates a dataset from Python lists/dicts.
    Useful for testing and creating small datasets programmatically.
    """
    data: dict[str, list[Any]]