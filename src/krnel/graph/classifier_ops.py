# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

from typing import Any, Literal
from pydantic import Field
from krnel.graph.op_spec import OpSpec
from krnel.graph.llm_ops import LLMLayerActivationsOp
from krnel.graph.types import BooleanColumnType, ClassifierType, ModelType, PreprocessingType, ScoreColumnType, TrainTestSplitColumnType, VectorColumnType, CategoricalColumnType


class TrainClassifierOp(ClassifierType):
    """
    An operation that trains a classifier model.
    """
    model_type: ModelType
    x: VectorColumnType
    y: BooleanColumnType

    train_domain: BooleanColumnType

    preprocessing: PreprocessingType = 'none'

    params: dict[str, Any] = Field(default_factory=dict)

class ClassifierPredictOp(ScoreColumnType):
    """
    An operation that performs prediction using a classifier model.
    """
    model: ClassifierType
    x: VectorColumnType