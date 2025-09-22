# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

from typing import Literal

from krnel.graph.dataset_ops import TextColumnType, VectorColumnType


class LLMGenerateTextOp(TextColumnType):
    model_name: str
    prompt: TextColumnType
    max_tokens: int = 100


class LLMLayerActivationsOp(VectorColumnType):
    model_name: str
    text: TextColumnType
    """
    The prompt to get activations for. We always apply chat template.
    """

    layer_num: int
    """Supports negative indexing: -1 = last layer, -2 = second-to-last.
    Not supported for SentenceTransformers or Ollama; set to -1 for those model providers.
    """

    token_mode: Literal["last", "mean", "all"]
    """Token pooling mode.  Not supported for Ollama or SentenceTransformers."""

    batch_size: int

    max_length: int | None = None
    "Maximum number of tokens in input. Longer prompts are truncated."

    dtype: str | None = None
    "DType of both the model itself and the output embeddings."

    device: str = "auto"
    "default: 'cuda' or 'mps' if available, else 'cpu'"

    torch_compile: bool = False
    "Whether to use torch.compile for performance optimization"
