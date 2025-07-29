from abc import ABC, abstractmethod
from typing import Any, Dict, Callable

import numpy as np
import httpx
from tqdm.auto import tqdm
from krnel.graph.llm_ops import LLMEmbedOp



class ModelProvider(ABC):
    """Abstract base class for model providers that can handle LLM operations."""

    @abstractmethod
    def embed(self, runner, op: LLMEmbedOp) -> np.ndarray:
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
            clean_scheme = scheme.rstrip(':')
            _PROVIDERS[clean_scheme] = instance
        return provider_class
    return decorator


def get_provider(model_url: str) -> tuple[ModelProvider, str]:
    """Get the provider and model name for a given model URL."""
    scheme, _, model_name = model_url.partition(':')

    if scheme not in _PROVIDERS:
        raise ValueError(f"No provider registered for scheme: {scheme}")

    return _PROVIDERS[scheme], model_name


def embed(runner, op: LLMEmbedOp) -> np.ndarray:
    """Dispatch embedding request to appropriate provider."""
    provider, model_name = get_provider(op.model_name)
    return provider.embed(runner, op_with_model)