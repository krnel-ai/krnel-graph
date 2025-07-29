import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from krnel.runners.model_registry import _PROVIDERS, get_provider, embed, OllamaProvider, TransformerLensProvider, HuggingFaceProvider
from krnel.runners.local_runner import LocalArrowRunner
from krnel.graph.llm_ops import LLMEmbedOp
from krnel.graph.dataset_ops import SelectTextColumnOp


class TestModelRegistry:
    def test_global_providers_exist(self):
        """Test that the global provider registry exists and has providers."""
        assert len(_PROVIDERS) > 0
    
    def test_decorators_registered_providers(self):
        """Test that decorators automatically registered providers."""
        expected_providers = ["ollama", "transformerlens", "tl", "huggingface", "hf"]
        for provider in expected_providers:
            assert provider in _PROVIDERS
    
    def test_url_parsing_scheme_format(self):
        """Test URL parsing with scheme:model format."""
        provider, model_name = get_provider("ollama:test-model")
        assert isinstance(provider, OllamaProvider)
        assert model_name == "test-model"
        
        # Test with different model name
        provider2, model_name2 = get_provider("ollama:test-model-2")
        assert isinstance(provider2, OllamaProvider)
        assert model_name2 == "test-model-2"
    
    def test_url_parsing_org_model_format(self):
        """Test URL parsing with scheme:org/model format."""
        provider, model_name = get_provider("hf:meta-llama/Llama-3.2-1B-Instruct")
        assert isinstance(provider, HuggingFaceProvider)
        assert model_name == "meta-llama/Llama-3.2-1B-Instruct"
    
    def test_unknown_provider_error(self):
        """Test error handling for unknown providers."""
        with pytest.raises(ValueError, match="No provider registered for scheme: unknown"):
            get_provider("unknown:model")


class TestGlobalModelProviders:
    def test_all_providers_preregistered(self):
        """Test that all providers are pre-registered via decorators."""
        expected_providers = ["ollama", "transformerlens", "tl", "huggingface", "hf"]
        for provider in expected_providers:
            assert provider in _PROVIDERS
            
        # Test Ollama
        provider, model_name = get_provider("ollama:test-model")
        assert isinstance(provider, OllamaProvider)
        assert model_name == "test-model"
        
        # Test TransformerLens
        provider, model_name = get_provider("tl:test-model")
        assert isinstance(provider, TransformerLensProvider)
        assert model_name == "test-model"
        
        # Test HuggingFace
        provider, model_name = get_provider("hf:test-model")
        assert isinstance(provider, HuggingFaceProvider)
        assert model_name == "test-model"


class TestProviders:
    def test_ollama_provider_initialization(self):
        """Test Ollama provider initialization."""
        provider = OllamaProvider()
        assert provider.server_url == "http://localhost:11434"
        
        custom_provider = OllamaProvider(server_url="http://custom:8080")
        assert custom_provider.server_url == "http://custom:8080"
    
    def test_transformerlens_provider_initialization(self):
        """Test TransformerLens provider initialization."""
        provider = TransformerLensProvider()
        assert provider._models == {}
        
    def test_huggingface_provider_initialization(self):
        """Test HuggingFace provider initialization."""
        provider = HuggingFaceProvider()
        assert provider._models == {}
        
    @patch('krnel.runners.model_registry.TORCH_AVAILABLE', True)
    @patch('krnel.runners.model_registry.torch')
    def test_device_detection(self, mock_torch):
        """Test device detection logic."""
        provider = OllamaProvider()  # Use any provider since method is inherited
        
        # Test explicit device
        assert provider._detect_device("cpu") == "cpu"
        assert provider._detect_device("cuda") == "cuda"
        
        # Test auto-detection
        mock_torch.cuda.is_available.return_value = True
        assert provider._detect_device("auto") == "cuda"
        
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        assert provider._detect_device("auto") == "mps"
        
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        assert provider._detect_device("auto") == "cpu"
    
    @patch('krnel.runners.model_registry.TORCH_AVAILABLE', False)
    def test_device_detection_no_torch(self):
        """Test device detection when torch is not available."""
        provider = OllamaProvider()
        
        # Should default to CPU when torch is not available
        assert provider._detect_device("auto") == "cpu"
        assert provider._detect_device("cpu") == "cpu"  # Explicit should still work


class TestTransformerLensProvider:
    @patch('krnel.runners.model_registry.TORCH_AVAILABLE', False)
    def test_torch_availability_check(self, tmp_path):
        """Test that TransformerLens provider checks for torch availability."""
        provider = TransformerLensProvider()
        runner = LocalArrowRunner(cache_folder=str(tmp_path))
        
        with pytest.raises(ImportError, match="torch is required for TransformerLensProvider"):
            dataset = runner.from_list({"text": ["test"]})
            text_column = SelectTextColumnOp(column_name="text", dataset=dataset)
            test_op = LLMEmbedOp(
                model_name="test-model",
                text=text_column,
                layer_num=-1,
                token_mode="last",
                max_length=2048,
                device="auto"
            )
            provider.embed(runner, test_op)
    
    @patch('krnel.runners.model_registry.TORCH_AVAILABLE', True)        
    def test_import_error_handling(self, tmp_path):
        """Test that TransformerLens provider handles missing imports gracefully."""
        provider = TransformerLensProvider()
        runner = LocalArrowRunner(cache_folder=str(tmp_path))
        
        with patch('builtins.__import__', side_effect=ImportError("No module named 'transformer_lens'")):
            with pytest.raises(ImportError, match="transformer_lens is required"):
                dataset = runner.from_list({"text": ["test"]})
                text_column = SelectTextColumnOp(column_name="text", dataset=dataset)
                test_op = LLMEmbedOp(
                    model_name="test-model",
                    text=text_column,
                    layer_num=-1,
                    token_mode="last",
                    max_length=2048,
                    device="auto"
                )
                provider.embed(runner, test_op)


class TestHuggingFaceProvider:
    @patch('krnel.runners.model_registry.TORCH_AVAILABLE', False)
    def test_torch_availability_check(self, tmp_path):
        """Test that HuggingFace provider checks for torch availability."""
        provider = HuggingFaceProvider()
        runner = LocalArrowRunner(cache_folder=str(tmp_path))
        
        with pytest.raises(ImportError, match="torch is required for HuggingFaceProvider"):
            dataset = runner.from_list({"text": ["test"]})
            text_column = SelectTextColumnOp(column_name="text", dataset=dataset)
            test_op = LLMEmbedOp(
                model_name="test-model",
                text=text_column,
                layer_num=-1,
                token_mode="last",
                max_length=2048,
                device="auto"
            )
            provider.embed(runner, test_op)
    
    @patch('krnel.runners.model_registry.TORCH_AVAILABLE', True)
    def test_import_error_handling(self, tmp_path):
        """Test that HuggingFace provider handles missing imports gracefully."""
        provider = HuggingFaceProvider()
        runner = LocalArrowRunner(cache_folder=str(tmp_path))
        
        with patch('builtins.__import__', side_effect=ImportError("No module named 'transformers'")):
            with pytest.raises(ImportError, match="transformers is required"):
                dataset = runner.from_list({"text": ["test"]})
                text_column = SelectTextColumnOp(column_name="text", dataset=dataset)
                test_op = LLMEmbedOp(
                    model_name="test-model",
                    text=text_column,
                    layer_num=-1,
                    token_mode="last",
                    max_length=2048,
                    device="auto"
                )
                provider.embed(runner, test_op)