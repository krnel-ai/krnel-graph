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
    
    @patch.object(OllamaProvider, 'embed')
    def test_registry_dispatch_new_format(self, mock_embed, tmp_path):
        """Test that new model URL format dispatches correctly to registry."""
        runner = LocalArrowRunner(cache_folder=str(tmp_path))
        
        # Mock the embed function
        mock_embeddings = np.random.random((5, 384))
        mock_embed.return_value = mock_embeddings
        
        # Create real data using from_list
        dataset = runner.from_list({"text": ["text1", "text2", "text3", "text4", "text5"]})
        text_column = SelectTextColumnOp(column_name="text", dataset=dataset)
        op = LLMEmbedOp(
            model_name="ollama:test-model", 
            text=text_column,
            layer_num=-1,
            token_mode="last",
            max_length=2048,
            device="auto"
        )
        
        # Get the implementation function and call it
        from krnel.runners.base_runner import _IMPLEMENTATIONS
        impl_func = _IMPLEMENTATIONS[LocalArrowRunner][LLMEmbedOp]
        result = impl_func(runner, op)
        
        # Verify embed was called once
        assert mock_embed.call_count == 1
        
        # Get the call arguments (now it's runner and LLMEmbedOp)
        call_args, call_kwargs = mock_embed.call_args
        call_runner, embed_op = call_args
        assert call_runner == runner
        assert embed_op.model_name == "test-model"  # URL should be parsed to extract model name
        assert embed_op.layer_num == -1
        assert embed_op.token_mode == "last"
        assert embed_op.max_length == 2048
        assert embed_op.device == "auto"
        
        # Verify result
        np.testing.assert_array_equal(result, mock_embeddings)
    
    @patch.object(OllamaProvider, 'embed')
    def test_backward_compatibility_old_format(self, mock_embed, tmp_path):
        """Test backward compatibility with old ollama:model format."""
        runner = LocalArrowRunner(cache_folder=str(tmp_path))
        
        # Mock the embed function
        mock_embeddings = np.random.random((3, 384))
        mock_embed.return_value = mock_embeddings
        
        # Create real data using from_list
        dataset = runner.from_list({"text": ["text1", "text2", "text3"]})
        text_column = SelectTextColumnOp(column_name="text", dataset=dataset)
        op = LLMEmbedOp(
            model_name="ollama:old-model",  # Old format that should still work
            text=text_column,
            layer_num=-1,
            token_mode="last",
            max_length=2048,
            device="auto"
        )
        
        # Get the implementation function and call it
        from krnel.runners.base_runner import _IMPLEMENTATIONS
        impl_func = _IMPLEMENTATIONS[LocalArrowRunner][LLMEmbedOp]
        result = impl_func(runner, op)
        
        # Verify embed was called with the old format parsed correctly
        call_args, call_kwargs = mock_embed.call_args
        call_runner, embed_op = call_args
        assert call_runner == runner
        assert embed_op.model_name == "old-model"  # Should be parsed from URL
        np.testing.assert_array_equal(result, mock_embeddings)
    
    @patch.object(TransformerLensProvider, 'embed')
    @patch.object(HuggingFaceProvider, 'embed')
    def test_new_providers_dispatch(self, mock_hf_embed, mock_tl_embed, tmp_path):
        """Test that new providers dispatch correctly."""
        runner = LocalArrowRunner(cache_folder=str(tmp_path))
        
        # Mock the embed functions
        mock_embeddings = np.random.random((3, 768))
        mock_tl_embed.return_value = mock_embeddings
        mock_hf_embed.return_value = mock_embeddings
        
        # Create real data using from_list
        dataset = runner.from_list({"text": ["text1", "text2", "text3"]})
        text_column = SelectTextColumnOp(column_name="text", dataset=dataset)
        op = LLMEmbedOp(
            model_name="tl:meta-llama/Llama-3.2-1B-Instruct",
            text=text_column,
            layer_num=-2,  # Second to last layer
            token_mode="mean",
            max_length=1024,
            device="cpu"
        )
        
        # Get the implementation function and call it
        from krnel.runners.base_runner import _IMPLEMENTATIONS
        impl_func = _IMPLEMENTATIONS[LocalArrowRunner][LLMEmbedOp]
        result = impl_func(runner, op)
        
        # Verify embed was called with correct arguments
        call_args, call_kwargs = mock_tl_embed.call_args
        call_runner, embed_op = call_args
        assert call_runner == runner
        assert embed_op.model_name == "meta-llama/Llama-3.2-1B-Instruct"  # URL should be parsed
        assert embed_op.layer_num == -2
        assert embed_op.token_mode == "mean"
        assert embed_op.max_length == 1024
        assert embed_op.device == "cpu"
        
        # Test result
        np.testing.assert_array_equal(result, mock_embeddings)
        
        # Reset and test HuggingFace provider
        mock_hf_embed.reset_mock()
        op_hf = LLMEmbedOp(
            model_name="hf:meta-llama/Llama-3.2-1B-Instruct",
            text=text_column,
            layer_num=-1,
            token_mode="last",
            max_length=2048,
            device="auto"
        )
        
        result_hf = impl_func(runner, op_hf)
        
        # Verify HuggingFace call
        call_args, call_kwargs = mock_hf_embed.call_args
        call_runner, embed_op = call_args
        assert call_runner == runner
        assert embed_op.model_name == "meta-llama/Llama-3.2-1B-Instruct"  # URL should be parsed
        assert embed_op.layer_num == -1
        assert embed_op.token_mode == "last"
        np.testing.assert_array_equal(result_hf, mock_embeddings)