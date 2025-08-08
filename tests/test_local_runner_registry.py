import pytest
import numpy as np
from unittest.mock import patch
from krnel.runners.local_runner import LocalArrowRunner
from krnel.runners.model_registry import OllamaProvider, TransformerLensProvider, HuggingFaceProvider
from krnel.graph.llm_ops import LLMEmbedOp
from krnel.graph.dataset_ops import SelectTextColumnOp


class TestLocalRunnerRegistryIntegration:
    def test_registry_llm_embed_implementation_exists(self, tmp_path):
        """Test that the registry_llm_embed implementation is registered."""
        runner = LocalArrowRunner(cache_folder=str(tmp_path))

        # Check that LLMEmbedOp is registered for LocalArrowRunner
        from krnel.runners.base_runner import _IMPLEMENTATIONS
        assert LLMEmbedOp in _IMPLEMENTATIONS[LocalArrowRunner]

        # Get the implementation function
        impl_func = _IMPLEMENTATIONS[LocalArrowRunner][LLMEmbedOp]
        assert impl_func.__name__ == "registry_llm_embed"
