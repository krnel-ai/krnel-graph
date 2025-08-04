from pydantic import SerializeAsAny
from krnel.graph.dataset_ops import JinjaTemplatizeOp, SelectColumnOp, TextColumnType, EmbeddingColumnType
from krnel.graph.op_spec import OpSpec


class LLMGenerateTextOp(OpSpec, TextColumnType):
    model_name: str
    prompt: TextColumnType
    max_tokens: int = 100

class LLMEmbedOp(OpSpec, EmbeddingColumnType):
    model_name: str
    text: TextColumnType
    layer_num: int
    token_mode: str