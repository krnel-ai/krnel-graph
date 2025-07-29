from krnel.graph.dataset_ops import JinjaTemplatizeOp, SelectColumnOp, TextColumnType, VectorColumnType
from krnel.graph.op_spec import OpSpec


class LLMGenerateTextOp(TextColumnType):
    model_name: str
    prompt: TextColumnType
    max_tokens: int = 100

class LLMEmbedOp(VectorColumnType):
    model_name: str
    text: TextColumnType
    layer_num: int  # Supports negative indexing: -1 = last layer, -2 = second-to-last
    token_mode: str  # "last", "mean", "all"