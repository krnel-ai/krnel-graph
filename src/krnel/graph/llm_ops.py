from pydantic import SerializeAsAny
from krnel.graph.dataset_ops import JinjaTemplatizeOp, SelectColumnOp, TextColumnType, VectorColumnType
from krnel.graph.op_spec import OpSpec


class LLMGenerateTextOp(OpSpec, TextColumnType):
    model_name: str
    prompt: SerializeAsAny[TextColumnType]
    max_tokens: int = 100

class LLMEmbedOp(OpSpec, VectorColumnType):
    model_name: str
    text: SerializeAsAny[TextColumnType]
    layer_num: int
    token_mode: str