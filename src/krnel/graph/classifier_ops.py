# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

from typing import Literal
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

    nu: float | None = None
    c: float | None = None
    gamma: float | None = None
    preprocessing: PreprocessingType = 'none'

class ClassifierPredictOp(ScoreColumnType):
    """
    An operation that performs prediction using a classifier model.
    """
    model: ClassifierType
    x: VectorColumnType