# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

from typing import Annotated, Any, ClassVar, Literal

from pydantic import BeforeValidator

from krnel.graph.op_spec import EphemeralOpMixin, ExcludeFromUUID, OpSpec
from krnel.graph.types import (
    BooleanColumnType,
    CategoricalColumnType,
    ConversationColumnType,
    DatasetType,
    RowIDColumnType,
    ScoreColumnType,
    TextColumnType,
    TrainTestSplitColumnType,
    VectorColumnType,
)

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


class LoadLocalParquetDatasetOp(LoadDatasetOp):
    file_path: Annotated[str, ExcludeFromUUID()]
    """Which file path this dataset was loaded from.

    Note:
      This path may not be accessible to remote runners. Calling ``Runner.prepare()`` on this op will copy this dataset into storage.
    """


class LoadInlineJsonDatasetOp(DatasetType):
    """
    An operation that creates a dataset from simple Python lists/dicts.
    Useful for testing and creating small datasets programmatically.

    Only dicts-of-lists (what pandas calls "columns" orientation) are supported.

    Example::

        dataset_op = LoadInlineJsonDatasetOp(data={
            "input": ["What is AI?", "What is the capital of France?"],
            "output": ["AI is ...", "The capital of France is Paris."],
            "embeddings": [[0.1, 0.2], [0.3, 0.4]],
            "is_legal": [True, False],
        })
    """

    data: dict[str, list[Any]]


class SelectColumnOp(OpSpec, EphemeralOpMixin):
    """
    A single column from the input dataset.
    """

    column_name: str
    dataset: DatasetType

    _op_func_name: ClassVar[str] = "col___FIXME___"

    def _code_repr_identifier(self, short=True) -> str:
        return "col_" + self.column_name + "_" + self.uuid_hash[:5]

    def _code_repr_statement(self) -> str:
        return f"{self._code_repr_identifier()} = {self.dataset._code_repr_expr()}.{self.__class__._op_func_name}({self.column_name!r})"

    def _repr_flowchart_node_(self):
        # return f"{self._code_repr_identifier()}[\"{self.column_name}\"]"
        return f'{self._code_repr_identifier()}@{{shape: "text", label: "{self.column_name}"}}'

    def _repr_flowchart_edges_(self):
        for dep in self.get_dependencies():
            yield f"{dep._code_repr_identifier()} --> {self._code_repr_identifier()}"


class SelectVectorColumnOp(SelectColumnOp, VectorColumnType):
    _op_func_name: ClassVar[str] = "col_vector"


class SelectTextColumnOp(SelectColumnOp, TextColumnType):
    _op_func_name: ClassVar[str] = "col_text"


class SelectConversationColumnOp(SelectColumnOp, ConversationColumnType):
    _op_func_name: ClassVar[str] = "col_conversation"


class SelectCategoricalColumnOp(SelectColumnOp, CategoricalColumnType):
    _op_func_name: ClassVar[str] = "col_categorical"


class SelectTrainTestSplitColumnOp(SelectColumnOp, TrainTestSplitColumnType):
    _op_func_name: ClassVar[str] = "col_train_test_split"


class SelectScoreColumnOp(SelectColumnOp, ScoreColumnType):
    _op_func_name: ClassVar[str] = "col_score"


class SelectBooleanColumnOp(SelectColumnOp, BooleanColumnType):
    _op_func_name: ClassVar[str] = "col_boolean"


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
    "If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples. If None, the value is set to the complement of the train size. If `train_size` is also None, it will be set to 0.25."

    train_size: float | int | None = None
    "If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split. If int, represents the absolute number of train samples. If None, the value is automatically set to the complement of the test size."

    random_state: int = 19190115
    "Defaults to date of the `Great Molasses Flood <https://en.wikipedia.org/wiki/Great_Molasses_Flood>`_."


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


def ensure_set_or_none(x):
    if x is not None:
        return sorted(set(x))
    return None


class CategoryToBooleanOp(BooleanColumnType, EphemeralOpMixin):
    """
    An operation that converts a categorical column to a boolean column.

    This is useful for binary classification tasks where the categorical
    values represent two distinct classes.

    When both `true_values` and `false_values` are provided,
    the set of actual values must be a subset
    of `true_values.union(false_values)`.

    When only `true_values` is provided, the operation will assume that all values not in `true_values` are false.

    When only `false_values` is provided, the operation will assume that all values not in `false_values` are true.
    """

    input_category: CategoricalColumnType | TrainTestSplitColumnType
    true_values: Annotated[list[str] | None, BeforeValidator(ensure_set_or_none)] = None
    false_values: Annotated[list[str] | None, BeforeValidator(ensure_set_or_none)] = (
        None
    )

    def _code_repr_statement(self) -> str | None:
        return None

    def _code_repr_expr(self) -> str:
        if self.true_values is not None and self.false_values is not None:
            return f"({self.input_category._code_repr_expr()}.is_in(true_values={self.true_values!r}, false_values={self.false_values!r}))"
        elif self.true_values is not None:
            return (
                f"{self.input_category._code_repr_expr()}.is_in({self.true_values!r})"
            )
        elif self.false_values is not None:
            return (
                f"{self.input_category._code_repr_expr()}.not_in({self.false_values!r})"
            )
        else:
            raise ValueError()

    def _repr_flowchart_node_(self):
        def _show(elts):
            if len(elts) > 5:
                return f"({len(elts)} choices)"
            return ", ".join(elts)

        results = []
        if self.true_values is not None:
            if len(self.true_values) == 1:
                results.append(f"= {self.true_values[0]}")
            else:
                results.append(f"∈ {{{_show(self.true_values)}}}")
        if self.false_values is not None:
            if len(self.false_values) == 1:
                results.append(f"≠ {self.false_values[0]}")
            else:
                results.append(f"∉ {{{_show(self.false_values)}}}")

        results = "; ".join(results)
        return f'{self._code_repr_identifier()}["{results}"]'

    def _repr_flowchart_edges_(self):
        for dep in self.get_dependencies():
            yield f"{dep._code_repr_identifier()} --> {self._code_repr_identifier()}"


class BooleanLogicOp(BooleanColumnType, EphemeralOpMixin):
    """An operation that carries out boolean operations.
    For the case of 'not', only the left column is used."""

    operation: Literal["and", "or", "xor", "not"]
    left: BooleanColumnType
    right: BooleanColumnType

    def _code_repr_statement(self) -> str | None:
        return None

    def _code_repr_expr(self) -> str:
        if self.operation == "not":
            return f"~ ({self.left._code_repr_expr()})"
        elif self.operation == "and":
            return f"({self.left._code_repr_expr()} & {self.right._code_repr_expr()})"
        elif self.operation == "or":
            return f"({self.left._code_repr_expr()} | {self.right._code_repr_expr()})"
        elif self.operation == "xor":
            return f"({self.left._code_repr_expr()} ^ {self.right._code_repr_expr()})"
        raise NotImplementedError()

    def _repr_flowchart_node_(self):
        return f'{self._code_repr_identifier()}{{"{self.operation.upper()}"}}'

    def _repr_flowchart_edges_(self):
        for dep in self.get_dependencies():
            yield f"{dep._code_repr_identifier()} --> {self._code_repr_identifier()}"
