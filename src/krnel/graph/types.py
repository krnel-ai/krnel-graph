# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

# Mixin types for various runtime objects
from typing import Any, Literal

from krnel.graph.op_spec import OpSpec

"""
Fluent API for operations across common types in the Krnel computation graph.

See the docstring of `dataset_ops.py` for more details.

"""


class DatasetType(OpSpec):
    """Base type for dataset operations in the computation graph.

    This class represents a dataset as a node in the OpSpec graph, providing
    methods to select and transform columns into specific typed operations.
    """

    def col_vector(self, column_name: str) -> "VectorColumnType":
        """Select a vector column from the dataset.

        Args:
            column_name: The name of the column containing vector embeddings.

        Returns:
            A VectorColumnType operation representing the selected embedding column.
        """
        from krnel.graph.dataset_ops import SelectVectorColumnOp

        return SelectVectorColumnOp(column_name=column_name, dataset=self)

    def col_text(self, column_name: str) -> "TextColumnType":
        """Select a text column from the dataset, typically containing prompts.

        Args:
            column_name: The name of the column containing text data.

        Returns:
            A TextColumnType operation representing the selected text column.
        """
        from krnel.graph.dataset_ops import SelectTextColumnOp

        return SelectTextColumnOp(column_name=column_name, dataset=self)

    def col_conversation(self, column_name: str) -> "ConversationColumnType":
        """Select a conversation column from the dataset, typically containing
        a list of {"role":...,"content":...} data.

        Args:
            column_name: The name of the column containing conversation data.

        Returns:
            A ConversationColumnType operation representing the selected column.
        """
        from krnel.graph.dataset_ops import SelectConversationColumnOp

        return SelectConversationColumnOp(column_name=column_name, dataset=self)

    def col_categorical(self, column_name: str) -> "CategoricalColumnType":
        """Select a categorical column from the dataset.

        Args:
            column_name: The name of the column containing categorical data.

        Returns:
            A CategoricalColumnType operation representing the selected categorical column.
        """
        from krnel.graph.dataset_ops import SelectCategoricalColumnOp

        return SelectCategoricalColumnOp(column_name=column_name, dataset=self)

    def col_train_test_split(self, column_name: str) -> "TrainTestSplitColumnType":
        """Select a train/test split column from the dataset.

        Args:
            column_name: The name of the column containing train/test split indicators.

        Returns:
            A TrainTestSplitColumnType operation representing the split column.
        """
        from krnel.graph.dataset_ops import SelectTrainTestSplitColumnOp

        return SelectTrainTestSplitColumnOp(column_name=column_name, dataset=self)

    def col_score(self, column_name: str) -> "ScoreColumnType":
        """Select a score column from the dataset.

        Args:
            column_name: The name of the column containing numerical scores or probabilities.

        Returns:
            A ScoreColumnType operation representing the selected score column.
        """
        from krnel.graph.dataset_ops import SelectScoreColumnOp

        return SelectScoreColumnOp(column_name=column_name, dataset=self)

    def col_boolean(self, column_name: str) -> "BooleanColumnType":
        """Select a boolean column from the dataset.

        Args:
            column_name: The name of the column containing true/false

        Returns:
            A BooleanColumnType operation representing the selected boolean column.
        """
        from krnel.graph.dataset_ops import SelectBooleanColumnOp

        return SelectBooleanColumnOp(column_name=column_name, dataset=self)

    def assign_train_test_split(
        self,
        test_size: float | int | None = None,
        train_size: float | int | None = None,
        random_state: int = 42,
    ) -> "TrainTestSplitColumnType":
        """Create a train/test split.

        Args:
            test_size: Size of the test set. Can be a float (proportion) or int (count).
                If None, will be inferred from train_size.
            train_size: Size of the training set. Can be a float (proportion) or int (count).
                If None, will be inferred from test_size.
            random_state: Random seed for reproducible splits.

        Returns:
            A TrainTestSplitColumnType operation representing the split assignment.
        """
        from krnel.graph.dataset_ops import AssignTrainTestSplitOp

        return AssignTrainTestSplitOp(
            dataset=self,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
        )

    def template(
        self, template: str, strip_template_whitespace=True, **context: "TextColumnType"
    ) -> "TextColumnType":
        """Apply a Jinja2 template to create new text content.

        Args:
            template: Jinja2 template string with placeholders.
            strip_template_whitespace: Whether to strip leading and trailing whitespace from the template input.
            **context: Named text columns to use as template variables.

        Returns:
            A TextColumnType operation with the templated text.

        Example::

            dataset.template(
                "Hello {{name}}, your score is {{score}}",
                name=dataset.col_prompt("name"),
                score=dataset.col_prompt("score")
            )
        """
        from krnel.graph.dataset_ops import JinjaTemplatizeOp

        if strip_template_whitespace:
            template = template.strip()
        return JinjaTemplatizeOp(template=template, context=context)

    def take(
        self, num_rows: int | None = None, *, skip: int = 1, offset: int = 0
    ) -> "DatasetType":
        """Sample rows from the dataset.

        Args:
            skip: Sampling interval - take every Nth row. Default is 1 (no skipping).
            offset: Number of rows to skip at the start before applying the skip pattern.
            num_rows: Maximum number of rows to return after applying the skip pattern. If None, returns all rows after applying the skip pattern.

        Returns:
            A new DatasetType operation with the sampled rows.

        Example:
            dataset.take(1000)  # Take first 1000 rows
            dataset.take(100, skip=10)  # Take every 10th row, up to 100 rows
        """
        from krnel.graph.dataset_ops import TakeRowsOp

        return TakeRowsOp(
            dataset=self,
            num_rows=num_rows,
            skip=skip,
            offset=offset,
            # content_hash=self.content_hash + f".take({num_rows}, {skip}, {offset})"
        )

    def mask_rows(self, mask: "BooleanColumnType") -> "DatasetType":
        """Filter rows in the dataset based on a boolean mask.

        Args:
            mask: A BooleanColumnType indicating which rows to keep (True) or discard (False).

        Returns:
            A new DatasetType operation with only the rows where the mask is True.
        """
        from krnel.graph.dataset_ops import MaskRowsOp

        return MaskRowsOp(dataset=self, mask=mask)

    def assign_row_id(self) -> "RowIDColumnType":
        """Assign a unique row ID to each row in the dataset.

        Returns:
            A new DatasetType operation with an additional column containing unique row IDs.
        """
        from krnel.graph.dataset_ops import AssignRowIDOp

        return AssignRowIDOp(dataset=self)


class RowIDColumnType(OpSpec):
    """Represents a column containing unique row IDs.

    This type is used to identify each row in the dataset uniquely, often
    created by the AssignRowIDOp operation.
    """

    ...


ModelType = Literal[
    "logistic_regression",
    "linear_svc",
    "rbf_svc",
    "rbf_nusvm",
    "calibrated_rbf_nusvm",
    "passive_aggressive",
]

PreprocessingType = Literal["none", "standardize", "normalize"]


class VectorColumnType(OpSpec):
    """Represents a column containing vector embeddings or numerical arrays.

    This type is used for operations that work with high-dimensional numerical
    data, such as embeddings from language models or other vector representations.
    """

    def train_classifier(
        self,
        model_type: ModelType,
        positives: "BooleanColumnType",
        negatives: "BooleanColumnType",
        train_domain: "BooleanColumnType | None" = None,
        preprocessing: PreprocessingType = "none",
        params: dict[str, Any] | None = None,
    ) -> "ClassifierType":
        """Train a classifier using this vector column as features.

        Args:
            model_type: Type of classifier model to train.
            labels: Categorical column containing the target labels.
            train_domain: Which samples to use for fitting, typically the training set.

        Returns:
            A ClassifierType operation representing the trained model.
        """
        if params is None:
            params = {}
        from krnel.graph.classifier_ops import TrainClassifierOp

        return TrainClassifierOp(
            model_type=model_type,
            x=self,
            positives=positives,
            negatives=negatives,
            train_domain=train_domain,
            preprocessing=preprocessing,
            params=params,
        )

    def umap_vis(
        self,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        n_epochs: int = 200,
        random_state: int = 42,
    ) -> "VizEmbeddingColumnType":
        """Create a 2D UMAP visualization of high-dimensional embeddings.

        Args:
            n_neighbors: Number of neighboring points used in local approximations
                of manifold structure. Larger values result in more global views
                of the manifold.
            min_dist: Minimum distance apart that points are allowed to be in
                the low dimensional representation.
            n_epochs: Number of training epochs for UMAP optimization.
            random_state: Random seed for reproducible results.

        Returns:
            A VizEmbeddingColumnType operation with 2D coordinates for visualization.
        """
        from krnel.graph.viz_ops import UMAPVizOp

        return UMAPVizOp(
            input_embedding=self,
            n_epochs=n_epochs,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
        )


class VizEmbeddingColumnType(OpSpec):
    """Represents a column containing 2D coordinates for visualization.

    This type is typically the result of dimensionality reduction operations
    like UMAP or t-SNE, containing x,y coordinates for plotting.
    """

    ...


class ClassifierType(OpSpec):
    """Represents a trained classification model.

    This type encapsulates a trained classifier that can be used to make
    predictions on new data.
    """

    def predict(self, input_data: "VectorColumnType") -> "ScoreColumnType":
        """Make predictions using the trained classifier.

        Args:
            input_data: Vector column containing the features to classify.

        Returns:
            A ScoreColumnType operation with the prediction scores/probabilities.
        """
        from krnel.graph.classifier_ops import ClassifierPredictOp

        return ClassifierPredictOp(
            model=self,
            x=input_data,
        )


class EvaluationReportType(OpSpec):
    """Represents a report containing evaluation metrics for a classifier model.

    This type is used to encapsulate the results of evaluating a classifier,
    including various metrics and visualizations.
    """

    ...


class TextColumnType(OpSpec):
    """Represents a column containing text data.

    This type is used for operations that work with textual data, such as
    prompts, generated text, or any string-based content.
    """

    def llm_generate_text(
        self, *, model_name: str, max_tokens: int = 100
    ) -> "TextColumnType":
        """Generate text using a language model.

        Args:
            model_name: Name/identifier of the language model to use.
            max_tokens: Maximum number of tokens to generate.

        Returns:
            A TextColumnType operation with the generated text.
        """
        from krnel.graph.llm_ops import LLMGenerateTextOp

        return LLMGenerateTextOp(
            model_name=model_name,
            prompt=self,
            max_tokens=max_tokens,
        )

    def llm_layer_activations(
        self,
        *,
        model_name: str,
        layer_num: int,
        token_mode: str,
        batch_size: int,
        dtype: str | None = None,
        max_length: int | None = None,
        device: str = "auto",
    ) -> VectorColumnType:
        """Extract layer activations from a language model.

        Args:
            model_name: Name/identifier of the language model to use.
            layer_num: Layer number to extract activations from. Supports negative
                indexing (-1 = last layer, -2 = second-to-last).
            token_mode: How to aggregate token activations. Options:
                - "last": Use the last token's activation
                - "mean": Average all token activations
                - "all": Return all token activations
            batch_size: Number of samples to process in each batch.
            dtype: Data type for model and output embeddings (e.g., "float32").
            max_length: Maximum sequence length to process. Longer sequences
                will be truncated.
            device: Device to run inference on. "auto" selects GPU if available.

        Returns:
            A VectorColumnType operation with the extracted activations.
        """
        from krnel.graph.llm_ops import LLMLayerActivationsOp

        return LLMLayerActivationsOp(
            model_name=model_name,
            text=self,
            layer_num=layer_num,
            token_mode=token_mode,
            dtype=dtype,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
        )


class ConversationColumnType(OpSpec):
    """Represents a column containing conversation or dialogue data.

    Example of one conversation:

    .. code-block:: python

        [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you! How can I assist you today?"},
            {"role": "user", "content": "What is the weather like today?"}
        ]

    This type is used for operations that work with structured conversational
    data, such as chat logs or dialogue datasets.
    """

    ...


class CategoricalColumnType(OpSpec):
    """Represents a column containing categorical (label) data.

    This type is used for operations that work with discrete categorical
    variables, such as class labels for classification tasks.
    """

    def is_in(
        self, true_values: set[str], *, false_values: set[str] | None = None
    ) -> "BooleanColumnType":
        from krnel.graph.dataset_ops import CategoryToBooleanOp

        return CategoryToBooleanOp(
            input_category=self, true_values=true_values, false_values=false_values
        )

    def not_in(self, false_values: set[str]) -> "BooleanColumnType":
        """Create a boolean column indicating rows not in the specified values.
        All other values are considered True.

        Args:
            false_values: Set of values that should be considered False.

        Returns:
            A BooleanColumnType operation where True indicates rows not in false_values.
        """
        from krnel.graph.dataset_ops import CategoryToBooleanOp

        return CategoryToBooleanOp(
            input_category=self, true_values=None, false_values=list(false_values)
        )


class TrainTestSplitColumnType(OpSpec):
    """Represents a column indicating train/test split assignments.

    This type contains boolean or categorical indicators specifying which
    rows belong to training vs testing sets in machine learning workflows.
    """

    def is_in(
        self, true_values: set[str] | None, *, false_values: set[str] | None = None
    ) -> "BooleanColumnType":
        from krnel.graph.dataset_ops import CategoryToBooleanOp

        return CategoryToBooleanOp(
            input_category=self,
            true_values=list(true_values) if true_values is not None else None,
            false_values=list(false_values) if false_values is not None else None,
        )


class ScoreColumnType(OpSpec):
    """Represents a column containing numerical scores or probabilities.

    This type is typically used for prediction scores, confidence values,
    or other numerical outputs from machine learning models.
    """

    def evaluate(
        self,
        gt_positives: "BooleanColumnType",
        gt_negatives: "BooleanColumnType",
        split: "TrainTestSplitColumnType | None" = None,
        predict_domain: "BooleanColumnType | None" = None,
    ) -> "EvaluationReportType":
        """Evaluate prediction scores against ground truth labels.

        Args:
            gt_positives: Boolean column with the true positive labels.
            gt_negatives: Boolean column with the true negative labels.
            split: Optional column indicating train/test split assignments.
              All metrics will be grouped by split.
            predict_domain: Optional column indicating which samples to include in evaluation.

        Returns:
            A ClassifierEvaluationOp operation with evaluation metrics.
        """
        from krnel.graph.classifier_ops import ClassifierEvaluationOp

        return ClassifierEvaluationOp(
            gt_positives=gt_positives,
            gt_negatives=gt_negatives,
            score=self,
            split=split,
            predict_domain=predict_domain,
        )


class BooleanColumnType(OpSpec):
    """Represents a column containing boolean values (True/False).

    This type is used for operations that require binary indicators or flags,
    such as filtering datasets based on certain conditions.
    """

    def __and__(self, other: "BooleanColumnType") -> "BooleanColumnType":
        "self & other"
        from krnel.graph.dataset_ops import BooleanLogicOp

        return BooleanLogicOp(operation="and", left=self, right=other)

    def __or__(self, other: "BooleanColumnType") -> "BooleanColumnType":
        "self | other"
        from krnel.graph.dataset_ops import BooleanLogicOp

        return BooleanLogicOp(operation="or", left=self, right=other)

    def __xor__(self, other: "BooleanColumnType") -> "BooleanColumnType":
        "self ^ other"
        from krnel.graph.dataset_ops import BooleanLogicOp

        return BooleanLogicOp(operation="xor", left=self, right=other)

    def __invert__(self) -> "BooleanColumnType":
        "~self"
        from krnel.graph.dataset_ops import BooleanLogicOp

        return BooleanLogicOp(operation="not", left=self, right=self)
