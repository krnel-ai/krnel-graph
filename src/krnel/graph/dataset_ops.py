from typing import Any, TypeVar, Generic
from krnel.graph import OpSpec
from krnel.graph.types import *


# Graph spec
class LoadDatasetOp(OpSpec, DatasetType):
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

class SelectPromptColumnOp(SelectColumnOp, TextColumnType):
    ...
class SelectTrainTestSplitColumnOp(SelectColumnOp, TrainTestSplitColumnType):
    ...
class SelectEmbeddingColumnOp(SelectColumnOp, VectorColumnType):
    ...
class SelectCategoricalColumnOp(SelectColumnOp, CategoricalColumnType):
    ...

class AssignTrainTestSplitOp(OpSpec, TrainTestSplitColumnType):
    """
    An operation that assigns a train/test split to a dataset column.
    """
    hash_column: TextColumnType
    test_size: float | int | None = None
    train_size: float | int | None = None
    random_state: int

class JinjaTemplatizeOp(OpSpec, TextColumnType):
    """
    An operation that templatizes a Jinja template with the given context.
    This can be used to create prompts, for example.
    """
    template: str
    context: dict[str, TextColumnType]