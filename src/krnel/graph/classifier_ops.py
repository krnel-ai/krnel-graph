from krnel.graph.op_spec import OpSpec
from krnel.graph.llm_ops import LLMEmbedOp
from krnel.graph.types import ClassifierType, ScoreColumnType, TrainTestSplitColumnType, VectorColumnType, CategoricalColumnType


class TrainClassifierOp(OpSpec, ClassifierType):
    """
    An operation that trains a classifier model.
    """
    model_name: str
    x: VectorColumnType
    y: CategoricalColumnType
    train_test_split: TrainTestSplitColumnType

class ClassifierPredictOp(OpSpec, ScoreColumnType):
    """
    An operation that performs prediction using a classifier model.
    """
    model: ClassifierType
    x: VectorColumnType