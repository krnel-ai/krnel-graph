# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

from krnel.graph.llm_ops import LLMLayerActivationsOp


class ModelProvider(ABC):
    """Abstract base class for model providers that can handle LLM operations."""

    @abstractmethod
    def get_layer_activations(self, runner, op: LLMLayerActivationsOp) -> np.ndarray:
        """Generate embeddings for the given LLMEmbedOp."""
        pass

    def _detect_device(self, device: str = "auto") -> str:
        """Auto-detect the best available device."""
        import torch

        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"


# Global provider registry
_PROVIDERS: Dict[str, ModelProvider] = {}


def register_model_provider(*schemes: str):
    """Decorator to register a provider class for one or more schemes."""

    def decorator(provider_class):
        instance = provider_class()
        for scheme in schemes:
            clean_scheme = scheme.rstrip(":")
            _PROVIDERS[clean_scheme] = instance
        return provider_class

    return decorator


def get_model_provider(model_url: str) -> tuple[ModelProvider, str]:
    """Get the provider and model name for a given model URL."""
    scheme, _, model_name = model_url.partition(":")

    if scheme not in _PROVIDERS:
        raise ValueError(f"No provider registered for scheme: {scheme}")

    return _PROVIDERS[scheme], model_name


def get_layer_activations(runner, op: LLMLayerActivationsOp) -> np.ndarray:
    """Dispatch embedding request to appropriate provider."""
    provider, model_name = get_model_provider(op.model_name)
    return provider.get_layer_activations(runner, op)
