# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

# ruff: noqa: S101, B017

"""Integration tests for embedding functionality using real models.

These tests perform end-to-end testing with actual model loading and inference,
using lightweight CPU-only models to ensure reliable testing without external dependencies.
"""

import numpy as np
import pytest

from krnel.graph.dataset_ops import SelectTextColumnOp
from krnel.graph.llm_ops import LLMLayerActivationsOp
from krnel.graph.runners import LocalArrowRunner


@pytest.fixture
def simple_texts():
    """Simple text samples for testing (avoiding chat template issues)."""
    return [
        "Hello world",
        "The quick brown fox",
        "Testing embeddings",
        "Short",
        "This is longer text for testing tokenization",
    ]


@pytest.fixture
def test_runner():
    return LocalArrowRunner(store_uri="memory://")


@pytest.fixture
def test_dataset(test_runner, simple_texts):
    """Test dataset created from simple texts."""
    return test_runner.from_inline_dataset({"text": simple_texts})


@pytest.fixture
def base_tl_embed_op(test_dataset):
    """Base TransformerLens embedding operation."""
    text_column = SelectTextColumnOp(column_name="text", dataset=test_dataset)
    return LLMLayerActivationsOp(
        model_name="tl:meta-llama/Llama-3.2-1B-Instruct",
        text=text_column,
        layer_num=-1,
        token_mode="last",
        batch_size=2,
        max_length=256,
        device="cpu",
        dtype="float32",
    )


@pytest.fixture
def base_hf_embed_op(test_dataset):
    """Base HuggingFace embedding operation."""
    text_column = SelectTextColumnOp(column_name="text", dataset=test_dataset)
    return LLMLayerActivationsOp(
        model_name="hf:Qwen/Qwen2.5-0.5B-Instruct",
        text=text_column,
        layer_num=-1,
        token_mode="last",
        batch_size=2,
        max_length=128,
        device="cpu",
        dtype="float32",
    )


@pytest.mark.ml_models
class TestTransformerLensBasic:
    """Test TransformerLens provider with models that have chat templates."""

    def test_llama_embeddings(self, test_runner, base_tl_embed_op):
        """Test basic TransformerLens embedding functionality with Llama."""
        # Generate embeddings
        result = test_runner.to_numpy(base_tl_embed_op)

        # Basic validation
        assert result.shape[0] == 5  # Number of input texts
        assert result.shape[1] == 2048  # Llama-3.2-1B hidden size
        assert result.dtype == np.float32

        # Check for reasonable values
        assert not np.allclose(result, 0)
        assert np.all(np.isfinite(result))

    def test_tl_consistency(self, test_runner, base_tl_embed_op):
        """Test TransformerLens embedding consistency."""
        # Generate embeddings twice
        result1 = test_runner.to_numpy(base_tl_embed_op)
        result2 = test_runner.to_numpy(base_tl_embed_op.subs(max_length=255))

        # Should be identical
        np.testing.assert_array_equal(result1, result2)

    def test_tl_token_modes(self, test_runner, base_tl_embed_op):
        """Test different token modes."""
        # Test last vs mean token mode
        embed_op_last = base_tl_embed_op.subs(token_mode="last")
        embed_op_mean = base_tl_embed_op.subs(token_mode="mean")

        result_last = test_runner.to_numpy(embed_op_last)
        result_mean = test_runner.to_numpy(embed_op_mean)

        # Should have same shape
        assert result_last.shape == result_mean.shape

        # Should be different (different pooling strategies)
        assert not np.allclose(result_last, result_mean)

    def test_tl_layer_selection(self, test_runner, base_tl_embed_op):
        """Test different layer selection."""
        # Test early vs last layer
        embed_op_early = base_tl_embed_op.subs(layer_num=5)
        embed_op_last = base_tl_embed_op.subs(layer_num=-1)

        result_early = test_runner.to_numpy(embed_op_early)
        result_last = test_runner.to_numpy(embed_op_last)

        # Should have same shape but different values
        assert result_early.shape == result_last.shape
        assert not np.allclose(result_early, result_last)

    def test_tl_requires_max_length(self, test_runner, base_tl_embed_op):
        """Test that TransformerLens requires max_length."""
        embed_op = base_tl_embed_op.subs(max_length=None)

        with pytest.raises(ValueError, match="TransformerLens requires max_length"):
            test_runner.to_numpy(embed_op)


@pytest.mark.ml_models
class TestHuggingFaceBasic:
    """Test HuggingFace provider with models that have chat templates."""

    def test_small_instruct_model(self, test_runner, base_hf_embed_op):
        """Test HuggingFace with a small instruction-tuned model."""
        result = test_runner.to_numpy(base_hf_embed_op)

        # Basic validation
        assert result.shape[0] == 5
        assert result.shape[1] > 0
        assert result.dtype == np.float32
        assert not np.allclose(result, 0)
        assert np.all(np.isfinite(result))

    def test_hf_consistency(self, test_runner, base_hf_embed_op):
        """Test HuggingFace consistency."""
        result1 = test_runner.to_numpy(base_hf_embed_op)
        result2 = test_runner.to_numpy(base_hf_embed_op.subs(max_length=127))

        np.testing.assert_array_equal(result1, result2)

    def test_hf_token_modes(self, test_runner, base_hf_embed_op):
        """Test HuggingFace token modes."""
        embed_op_last = base_hf_embed_op.subs(token_mode="last")
        embed_op_mean = base_hf_embed_op.subs(token_mode="mean")

        result_last = test_runner.to_numpy(embed_op_last)
        result_mean = test_runner.to_numpy(embed_op_mean)

        assert result_last.shape == result_mean.shape
        assert not np.allclose(result_last, result_mean)

    def test_hf_requires_max_length(self, test_runner, base_hf_embed_op):
        """Test that HuggingFace requires max_length."""
        embed_op = base_hf_embed_op.subs(max_length=None)

        with pytest.raises(ValueError, match="HuggingFace requires max_length"):
            test_runner.to_numpy(embed_op)

    def test_hf_invalid_token_mode(self, test_runner, base_hf_embed_op):
        """Test invalid token mode handling."""
        embed_op = base_hf_embed_op.subs(
            token_mode="all"
        )  # Not supported by HF provider

        with pytest.raises(ValueError, match="Unsupported token_mode for HuggingFace"):
            test_runner.to_numpy(embed_op)


@pytest.mark.ml_models
class TestOllamaIntegration:
    """Test Ollama provider (conditional on server availability)."""

    def test_ollama_server_availability(self, test_runner, test_dataset):
        """Test Ollama integration if server is available, otherwise skip."""
        pytest.importorskip("httpx")
        import httpx

        try:
            # Quick check if Ollama server is running
            response = httpx.get("http://localhost:11434/api/version", timeout=5.0)
            if response.status_code != 200:
                pytest.skip("Ollama server not available")
        except Exception:
            pytest.skip("Ollama server not available")

        # If we get here, server is available - test with a simple model
        text_column = SelectTextColumnOp(column_name="text", dataset=test_dataset)
        embed_op = LLMLayerActivationsOp(
            model_name="ollama:all-minilm",  # Assumes this model is available
            text=text_column,
            layer_num=-1,
            token_mode="last",
            batch_size=2,
            max_length=None,
            device="cpu",
            dtype=None,
        )

        try:
            result = test_runner.to_numpy(embed_op)

            # Basic validation
            assert result.shape[0] == 5
            assert len(result.shape) == 2  # Should be 2D
            assert not np.allclose(result, 0)
            assert np.all(np.isfinite(result))
        except Exception as e:
            if "model" in str(e).lower() and "not found" in str(e).lower():
                pytest.skip(f"Required Ollama model not available: {e}")
            else:
                raise


@pytest.mark.ml_models
class TestErrorHandling:
    """Test error handling across providers."""

    def test_invalid_model_name(self, test_runner, base_tl_embed_op):
        """Test handling of invalid model names."""
        embed_op = base_tl_embed_op.subs(model_name="tl:nonexistent-model-12345")

        # Should raise some kind of error (model not found, etc.)
        with pytest.raises(Exception):
            test_runner.to_numpy(embed_op)

    def test_invalid_provider_scheme(self, test_runner, base_tl_embed_op):
        """Test handling of invalid provider schemes."""
        embed_op = base_tl_embed_op.subs(model_name="nonexistent:some-model")

        with pytest.raises(ValueError, match="No provider registered for scheme"):
            test_runner.to_numpy(embed_op)

    def test_tl_invalid_layer_num(self, test_runner, base_tl_embed_op):
        """Test TransformerLens layer number validation."""
        # Llama-3.2-1B has 16 layers, so layer 50 should be invalid
        embed_op = base_tl_embed_op.subs(layer_num=50)

        with pytest.raises(ValueError, match="layer_num.*out of range"):
            test_runner.to_numpy(embed_op)


@pytest.mark.ml_models
class TestBatchProcessing:
    """Test batch processing consistency."""

    def test_different_batch_sizes(self, test_runner, base_tl_embed_op):
        """Test that different batch sizes produce same embeddings."""
        embed_op1 = base_tl_embed_op.subs(batch_size=1)  # Process one at a time
        embed_op2 = base_tl_embed_op.subs(batch_size=5)  # Process all at once

        result1 = test_runner.to_numpy(embed_op1)
        result2 = test_runner.to_numpy(embed_op2)

        # Should be very close (allow small numerical differences)
        np.testing.assert_allclose(result1, result2, rtol=1e-4, atol=1e-6)
