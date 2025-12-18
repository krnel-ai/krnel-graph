# Copyright (c) 2025-2026 Krnel
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
        """
        from krnel.graph.dataset_ops import SelectVectorColumnOp

        return SelectVectorColumnOp(column_name=column_name, dataset=self)

    def col_text(self, column_name: str) -> "TextColumnType":
        """Select a text column from the dataset, typically containing prompts.

        Args:
            column_name: The name of the column containing text data.
        """
        from krnel.graph.dataset_ops import SelectTextColumnOp

        return SelectTextColumnOp(column_name=column_name, dataset=self)

    def col_conversation(self, column_name: str) -> "ConversationColumnType":
        """Select a conversation column from the dataset, typically containing
        a list of {"role":...,"content":...} data.

        Args:
            column_name: The name of the column containing conversation data.
        """
        from krnel.graph.dataset_ops import SelectConversationColumnOp

        return SelectConversationColumnOp(column_name=column_name, dataset=self)

    def col_categorical(self, column_name: str) -> "CategoricalColumnType":
        """Select a categorical column from the dataset.

        Args:
            column_name: The name of the column containing categorical data.
        """
        from krnel.graph.dataset_ops import SelectCategoricalColumnOp

        return SelectCategoricalColumnOp(column_name=column_name, dataset=self)

    def col_train_test_split(self, column_name: str) -> "TrainTestSplitColumnType":
        """Select a train/test split column from the dataset.

        Args:
            column_name: The name of the column containing train/test split indicators.
        """
        from krnel.graph.dataset_ops import SelectTrainTestSplitColumnOp

        return SelectTrainTestSplitColumnOp(column_name=column_name, dataset=self)

    def col_score(self, column_name: str) -> "ScoreColumnType":
        """Select a score column from the dataset.

        Args:
            column_name: The name of the column containing numerical scores or probabilities.
        """
        from krnel.graph.dataset_ops import SelectScoreColumnOp

        return SelectScoreColumnOp(column_name=column_name, dataset=self)

    def col_boolean(self, column_name: str) -> "BooleanColumnType":
        """Select a boolean column from the dataset.

        Args:
            column_name: The name of the column containing true/false
        """
        from krnel.graph.dataset_ops import SelectBooleanColumnOp

        return SelectBooleanColumnOp(column_name=column_name, dataset=self)

    def col_json(self, column_name: str) -> "JSONColumnType":
        """Select a JSON column from the dataset.

        Args:
            column_name: The name of the column containing JSON-serializable structured data.
        """
        from krnel.graph.dataset_ops import SelectJSONColumnOp

        return SelectJSONColumnOp(column_name=column_name, dataset=self)

    def assign_train_test_split(
        self,
        test_size: float | int | None = None,
        train_size: float | int | None = None,
        random_state: int = 42,
    ) -> "TrainTestSplitColumnType":
        """Assigns a new train/test split to this dataset.

        To load the train/test split from an existing column in the database, use
        :meth:`DatasetType.col_train_test_split() <krnel.graph.types.DatasetType.col_train_test_split>` instead.

        Args:
            test_size: Size of the test set. Can be a float (proportion between 0.0 and 1.0) or int (absolute count of samples).
                If None, will be inferred from train_size.
                If train_size is also None, it will be set to 0.25.
            train_size: Size of the training set. Can be a float (proportion between 0.0 and 1.0) or int (absolute count of samples).
                If None, will be set to the complement of test_size.
            random_state: Random seed for reproducible splits.
        """
        from krnel.graph.dataset_ops import AssignTrainTestSplitOp

        return AssignTrainTestSplitOp(
            dataset=self,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
        )

    def template(
        self,
        template: str,
        strip_template_whitespace=True,
        max_length: int | None = None,
        **context: "TextColumnType | JSONColumnType | float | int | str",
    ) -> "TextColumnType":
        """Apply a Jinja2 template to create new text content.

        Args:
            template: Jinja2 template string with placeholders.
            strip_template_whitespace: Whether to strip leading and trailing whitespace from the template input.
            max_length: Maximum length of the output text. If None, no truncation is applied.
            **context: Named text columns, JSON columns, or scalar values to use as template variables.

        Example:
            ::

                dataset.template(
                    "Hello {{name}}, your score is {{score}}. Max score: {{max_score}}",
                    name=dataset.col_text("name"),
                    score=dataset.col_text("score"),
                    max_score=100,
                )
        """
        from krnel.graph.dataset_ops import JinjaTemplatizeOp

        # Move constants into a different parameter.
        extra_context: dict[str, float | int | str] = {}
        column_context: dict[str, TextColumnType | JSONColumnType] = {}
        for k, v in context.items():
            if isinstance(v, (TextColumnType, JSONColumnType)):
                column_context[k] = v
            else:
                extra_context[k] = v  # type: ignore[assignment]

        if strip_template_whitespace:
            template = template.strip()
        return JinjaTemplatizeOp(template=template, context=column_context, constants=extra_context, max_length=max_length)
    def take(
        self, num_rows: int | None = None, *, skip: int = 1, offset: int = 0
    ) -> "DatasetType":
        """Sample rows from the dataset.

        Args:
            skip: Sampling interval - take every Nth row. Default is 1 (no skipping).
            offset: Number of rows to skip at the start before applying the skip pattern.
            num_rows: Maximum number of rows to return after applying the skip pattern. If None, returns all rows after applying the skip pattern.

        Example:
            ::

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

        .. topic:: Graph operation

            This method is implemented by a :obj:`krnel.graph.dataset_ops.MaskRowsOp` in the graph.

        Args:
            mask: A BooleanColumnType indicating which rows to keep (True) or discard (False).

        Example:
            .. code-block::

                import krnel.graph as kg
                runner = kg.Runner()
                ds = runner.from_parquet("foo.parquet")
                col_is_valid = ds.col_boolean("is_valid")

                masked_dset = ds.mask_rows(col_is_valid)

                # quick sample of 10% of the rows
                mask = ds.assign_train_test_split(test_size=0.1).test
                masked_dset = ds.mask_rows(mask)

        """
        from krnel.graph.dataset_ops import MaskRowsOp

        return MaskRowsOp(dataset=self, mask=mask)

    # def assign_row_id(self) -> "RowIDColumnType":
    #    """Assign a unique row ID to each row in the dataset.

    #    Returns:
    #        A new DatasetType operation with an additional column containing unique row IDs.
    #    """
    #    from krnel.graph.dataset_ops import AssignRowIDOp

    #    return AssignRowIDOp(dataset=self)


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
        *,
        positives: "BooleanColumnType",
        negatives: "BooleanColumnType | None" = None,
        train_domain: "BooleanColumnType | None" = None,
        preprocessing: PreprocessingType = "none",
        params: dict[str, Any] | None = None,
    ) -> "ClassifierType":
        """Train a classifier using this vector column as features.

        Args:
            model_type: Type of classifier model to train.
            positives: Boolean column indicating positive class samples.
            negatives: Boolean column indicating negative class samples. If None, negatives are the inverse of positives. Samples that are neither positive nor negative are ignored.
            train_domain: Which samples to use for fitting, typically the training set.
            preprocessing: Which preprocessing to apply to vectors.

               - ``none``: No preprocessing.
               - ``standardize``: Standardize features by removing the mean and scaling to unit variance.
               - ``normalize``: Scale input vectors to unit norm (vector length = 1).
            params: Additional model-specific :external+sklearn:term:`parameters for scikit-learn estimators <parameter>`, e.g. ``{"C": 0.01}``.

        See Also:

            Represented by :obj:`TrainClassifierOp <krnel.graph.classifier_ops.TrainClassifierOp>`



        """
        if params is None:
            params = {}
        from krnel.graph.classifier_ops import TrainClassifierOp

        if negatives is None:
            negatives = ~positives

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

        """
        from krnel.graph.viz_ops import UMAPVizOp

        return UMAPVizOp(
            input_embedding=self,
            n_epochs=n_epochs,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
        )

    def col(self, index: int) -> "ScoreColumnType":
        """Extract a specific dimension from the vector column as a scalar column.

        Args:
            index: The index of the dimension to extract.
        """
        from krnel.graph.dataset_ops import VectorToScalarOp

        return VectorToScalarOp(input=self, col_index=index)


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

        .. warning:: Currently unimplemented.

        Args:
            model_name: Name/identifier of the language model to use.
            max_tokens: Maximum number of tokens to generate.

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
        apply_chat_template: bool = True,
    ) -> VectorColumnType:
        """Extract layer activations from a language model. Can plug into several inference frameworks, e.g. ollama, transformers, transformer-lens, ...

        This variant discards logits and only extracts one layer's activations at a time. To capture logits, instead use :meth:`llm_logit_scores() <krnel.graph.types.TextColumnType.llm_logit_scores>`.

        Example:
            ::

                import krnel.graph as kg
                runner = kg.Runner()
                ds = runner.from_parquet("dataset.parquet")
                col_text = ds.col_text("prompt_col_in_pq_file")

                X = col_text.llm_layer_activations(
                    model_name="hf:meta-llama/Llama-2-7b-chat-hf",
                    layer_num=-1,
                    token_mode="last",
                    batch_size=4,
                    max_length=2048,
                    dtype="float16",
                )

                print(X.to_numpy().shape)  # e.g., (num_samples, embedding_dim)

        Args:
            model_name: Name/identifier of the language model to use. Three schemes are supported:

                Huggingface Transformers models:
                    - Prefix with ``hf:`` (e.g., ``"hf:meta-llama/Llama-2-7b-chat-hf"``)

                TransformerLens models:
                    - Prefix with ``tl:`` (e.g., ``"tl:gpt2-small"``)

                Ollama models:
                    - Prefix with ``ollama:`` (e.g., ``ollama:llama2``)
                    - Requires a running Ollama server on `localhost`.
                    - Only certain older Ollama models are supported, and only the last layer (``layer_num=-1``) can be extracted. Ollama is transitioning to a new Go-based inference engine that does not currently support getting layer activations; see `issue #10824 <https://github.com/ollama/ollama/issues/10824>`_.

            layer_num: Layer number to extract activations from. Supports negative
                indexing (-1 = last layer, -2 = second-to-last).
            token_mode: How to aggregate token activations. Options:

                - ``"last"``: Use the last token's activation
                - ``"mean"``: Average all token activations
                - ``"all"``: Return all token activations

            batch_size: Number of samples to process in each batch.
            dtype: Data type for model and output embeddings (e.g., "float32").
            max_length: Maximum sequence length to process. Longer sequences
                will be truncated.
            device: Device to run inference on. "auto" selects GPU if available.
            apply_chat_template: Whether to apply the model's chat template to format
                the input text as a conversation. Default is ``True``. Set to ``False``
                to tokenize raw text directly without conversation formatting.

        Returns:
            An operation that computes the extracted activations as a vector column.


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
            apply_chat_template=apply_chat_template,
        )

    def llm_logit_scores(
        self,
        *,
        model_name: str,
        logit_token_ids: list[str | int],
        batch_size: int,
        max_length: int,
        apply_chat_template: bool = True,
        dtype: str | None = None,
        device: str = "auto",
        torch_compile: bool = False,
    ) -> VectorColumnType:
        """Get logit scores for specified tokens from a language model. Useful for guardrailing.

        Args:
            model_name: Name/identifier of the language model to use.
            logit_token_ids: List of tokens (strings or token IDs) to get logit scores for.
            batch_size: Number of samples to process in each batch.
            max_length: Maximum sequence length to process. Longer sequences will be truncated.
            apply_chat_template: Whether to apply the model's chat template to format
                the input text as a conversation. Default is ``True``. Set to ``False``
                to tokenize raw text directly without conversation formatting.
            dtype: Data type for model and output scores (e.g., "float32").
            device: Device to run inference on. "auto" selects MPS or CUDA if available.
            torch_compile: Whether to use torch.compile for performance optimization.

        Returns:
            A VectorColumnType operation with the logit scores for the specified tokens.
        """
        from krnel.graph.llm_ops import LLMLogitScoresOp

        return LLMLogitScoresOp(
            model_name=model_name,
            text=self,
            logit_token_ids=logit_token_ids,
            batch_size=batch_size,
            apply_chat_template=apply_chat_template,
            dtype=dtype,
            device=device,
            max_length=max_length,
            torch_compile=torch_compile,
        )

    def parse_json(self) -> "JSONColumnType":
        """Parse JSON strings from this text column into structured JSON data.

        Returns:
            A JSONColumnType operation containing the parsed JSON structures.

        Example:
            ::

                import krnel.graph as kg
                runner = kg.Runner()
                ds = runner.from_parquet("dataset.parquet")
                json_text = ds.col_text("json_string_column")
                json_data = json_text.parse_json()
        """
        from krnel.graph.dataset_ops import ParseJSONColumnOp

        return ParseJSONColumnOp(text=self)

    def is_in(
        self, true_values: set[str], *, false_values: set[str] | None = None
    ) -> "BooleanColumnType":
        from krnel.graph.dataset_ops import CategoryToBooleanOp

        return CategoryToBooleanOp(
           input_category=self, true_values=list(true_values), false_values=list(false_values) if false_values is not None else None
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


class JSONColumnType(OpSpec):
    """Represents a column containing JSON-serializable structured data.

    This type is used for operations that work with structured data that may or
    may not conform to a specific schema. JSON columns can store nested
    dictionaries, lists, and other JSON-serializable Python structures.

    Examples of JSON column data:

    Agent conversations:
        .. code-block:: python

            [
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing well!"}
            ]

    Tool calls and results:
        .. code-block:: python

            {
                "tool": "search",
                "args": {"query": "weather today"},
                "result": {"temperature": 72, "condition": "sunny"}
            }

    Nested structures:
        .. code-block:: python

            {
                "metadata": {"version": 1, "source": "api"},
                "data": [{"id": 1, "value": "test"}],
                "tags": ["important", "urgent"]
            }
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

    @property
    def train(self) -> "BooleanColumnType":
        """Boolean column indicating training set membership."""
        from krnel.graph.dataset_ops import CategoryToBooleanOp
        return CategoryToBooleanOp(input_category=self, true_values=["train"])
    @property
    def test(self) -> "BooleanColumnType":
        """Boolean column indicating training set membership."""
        from krnel.graph.dataset_ops import CategoryToBooleanOp
        return CategoryToBooleanOp(input_category=self, true_values=["test"])


class ScoreColumnType(OpSpec):
    """Represents a column containing numerical scores or probabilities.

    This type is typically used for prediction scores, confidence values,
    or other numerical outputs from machine learning models.
    """

    def evaluate(
        self,
        gt_positives: "BooleanColumnType",
        gt_negatives: "BooleanColumnType | None" = None,
        split: "TrainTestSplitColumnType | None" = None,
        predict_domain: "BooleanColumnType | None" = None,
        score_threshold: float | None = None,
    ) -> "EvaluationReportType":
        """Evaluate prediction scores against ground truth labels.

        Args:
            gt_positives: Boolean column with the true positive labels.
            gt_negatives: Boolean column with the true negative labels. If None, negatives are the inverse of gt_positives. Samples that are neither positive nor negative are ignored.
            split: Optional column indicating train/test split assignments.
              All metrics will be grouped by split.
            predict_domain: Optional column indicating which samples to include in evaluation.
            score_threshold: Optional threshold to binarize scores into predictions. (If None, will pick the threshold that maximizes accuracy. If None and all labels are equal, then accuracy will be NaN.)

        """
        from krnel.graph.classifier_ops import ClassifierEvaluationOp

        if gt_negatives is None:
            gt_negatives = ~gt_positives

        return ClassifierEvaluationOp(
            gt_positives=gt_positives,
            gt_negatives=gt_negatives,
            score=self,
            split=split,
            predict_domain=predict_domain,
            score_threshold=score_threshold,
        )

    def __add__(self, other: "ScoreColumnType") -> "ScoreColumnType":
        "self + other"
        from krnel.graph.dataset_ops import PairwiseArithmeticOp
        if not isinstance(other, ScoreColumnType):
            raise ValueError("Can only add ScoreColumnType to ScoreColumnType")
        return PairwiseArithmeticOp(operation="+", left=self, right=other)
    def __sub__(self, other: "ScoreColumnType") -> "ScoreColumnType":
        "self - other"
        from krnel.graph.dataset_ops import PairwiseArithmeticOp
        if not isinstance(other, ScoreColumnType):
            raise ValueError("Can only subtract ScoreColumnType from ScoreColumnType")
        return PairwiseArithmeticOp(operation="-", left=self, right=other)
    def __mul__(self, other: "ScoreColumnType") -> "ScoreColumnType":
        "self * other"
        from krnel.graph.dataset_ops import PairwiseArithmeticOp
        if not isinstance(other, ScoreColumnType):
            raise ValueError("Can only multiply ScoreColumnType by ScoreColumnType")
        return PairwiseArithmeticOp(operation="*", left=self, right=other)
    def __truediv__(self, other: "ScoreColumnType") -> "ScoreColumnType":
        "self / other"
        from krnel.graph.dataset_ops import PairwiseArithmeticOp
        if not isinstance(other, ScoreColumnType):
            raise ValueError("Can only divide ScoreColumnType by ScoreColumnType")
        return PairwiseArithmeticOp(operation="/", left=self, right=other)


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
