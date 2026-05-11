# Copyright (c) 2025-2026 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

from typing import Any

from pydantic import Field

from krnel.graph.repr_html import FlowchartBigNode
from krnel.graph.types import (
    BooleanColumnType,
    CategoricalColumnType,
    ClassifierType,
    EvaluationReportType,
    PreprocessingType,
    ScoreColumnType,
    TrainTestSplitColumnType,
    VectorColumnType,
)


class TrainClassifierOp(FlowchartBigNode, ClassifierType):
    """
    An operation that trains a classifier model.
    """

    model_type: str
    "Classifier algorithm. Currently supported: ``'logistic_regression'``."

    x: VectorColumnType
    "Input feature vectors to train on."

    positives: BooleanColumnType
    "Boolean mask selecting rows that are positive examples."

    negatives: BooleanColumnType
    "Boolean mask selecting rows that are negative examples."

    train_domain: BooleanColumnType | None
    "Optional mask further restricting which rows are used for training. ``None`` uses all rows selected by ``positives``/``negatives``."

    preprocessing: PreprocessingType = "none"
    "Feature preprocessing applied before training. One of ``'none'``, ``'standard_scaler'``."

    params: dict[str, Any] = Field(default_factory=dict)
    "Additional hyperparameters forwarded to the underlying sklearn estimator (e.g. ``{'C': 0.1}`` for logistic regression)."


class ClassifierPredictOp(FlowchartBigNode, ScoreColumnType):
    """
    An operation that performs prediction using a classifier model.
    """

    model: ClassifierType
    "The trained classifier to use for prediction."

    x: VectorColumnType
    "Input feature vectors to score. Typically produced by the same op as the training vectors, with a different dataset substituted via ``subs()``."


class ClassifierEvaluationOp(FlowchartBigNode, EvaluationReportType):
    """
    An operation that evaluates prediction scores.

    Metrics and results are binned by each split (training, testing, etc)
    """

    score: ScoreColumnType
    "Predicted scores to evaluate, produced by :class:`ClassifierPredictOp`."

    gt_positives: BooleanColumnType
    "Ground-truth boolean mask for positive examples."

    gt_negatives: BooleanColumnType
    "Ground-truth boolean mask for negative examples."

    split: TrainTestSplitColumnType | CategoricalColumnType | None
    "Optional column to bin metrics by (e.g. a train/test split). ``None`` reports aggregate metrics only."

    predict_domain: BooleanColumnType | None
    "Optional mask restricting which rows are included in evaluation. ``None`` evaluates all rows."

    score_threshold: float | None = None
    "Optional threshold to binarize scores into predictions. (If None, will pick the threshold that maximizes accuracy. If None and all labels are equal, then accuracy will be NaN.)"