# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

from typing import Any

from pydantic import Field

from krnel.graph.repr_html import FlowchartBigNode
from krnel.graph.types import (
    BooleanColumnType,
    ClassifierType,
    EvaluationReportType,
    ModelType,
    PreprocessingType,
    ScoreColumnType,
    TrainTestSplitColumnType,
    VectorColumnType,
)


class TrainClassifierOp(FlowchartBigNode, ClassifierType):
    """
    An operation that trains a classifier model.
    """

    model_type: ModelType
    x: VectorColumnType

    positives: BooleanColumnType
    negatives: BooleanColumnType
    train_domain: BooleanColumnType | None

    preprocessing: PreprocessingType = "none"

    params: dict[str, Any] = Field(default_factory=dict)


class ClassifierPredictOp(FlowchartBigNode, ScoreColumnType):
    """
    An operation that performs prediction using a classifier model.
    """

    model: ClassifierType
    x: VectorColumnType


class ClassifierEvaluationOp(FlowchartBigNode, EvaluationReportType):
    """
    An operation that evaluates prediction scores.

    Metrics and results are binned by each split (training, testing, etc)
    """

    score: ScoreColumnType
    gt_positives: BooleanColumnType
    gt_negatives: BooleanColumnType
    split: TrainTestSplitColumnType | None

    predict_domain: BooleanColumnType | None
