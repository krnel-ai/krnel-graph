# Copyright (c) 2025 Krnel
# Points of Contact: 
#   - kimmy@krnel.ai

from krnel.graph.dataset_ops import JinjaTemplatizeOp, SelectColumnOp, TextColumnType, VectorColumnType
from krnel.graph.op_spec import OpSpec


class LLMGenerateTextOp(TextColumnType):
    model_name: str
    prompt: TextColumnType
    max_tokens: int = 100

class LLMLayerActivationsOp(VectorColumnType):
    model_name: str
    text: TextColumnType
    layer_num: int  # Supports negative indexing: -1 = last layer, -2 = second-to-last
    token_mode: str  # "last", "mean", "all"
    batch_size: int

    max_length: int | None = None
    dtype: str | None = None  # DType of both the model itself and the output embeddings.
    device: str = "auto" # default: "cuda" or "mps" if available, else "cpu"
