# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai


import httpx
import numpy as np
from tqdm.auto import tqdm

from krnel.graph.llm_ops import LLMLayerActivationsOp
from krnel.graph.runners.model_registry import (
    ModelProvider,
    get_model_provider,
    register_model_provider,
)
from krnel.logging import get_logger

logger = get_logger(__name__)


@register_model_provider("ollama")
class OllamaProvider(ModelProvider):
    """Provider for Ollama local LLM server."""

    def __init__(
        self, server_url: str = "http://localhost:11434", timeout: float = 60.0
    ):
        self.server_url = server_url
        self.timeout = timeout

    def get_layer_activations(self, runner, op: LLMLayerActivationsOp) -> np.ndarray:
        """Generate embeddings using Ollama API."""
        _, model_name = get_model_provider(op.model_name)
        log = logger.bind(model_name=model_name, op=op.uuid)
        if op.layer_num != -1:
            raise ValueError(
                "Ollama does not support layer_num; it always returns the last layer."
            )
        if op.token_mode != "last":
            raise ValueError("Ollama only supports 'last' token mode for embeddings.")
        # assert op.dtype is None, "Ollama does not support dtype specification."
        # assert op.max_length is None, "Configuring max_length is not supported for Ollama."

        # Materialize the text data
        texts = runner.to_numpy(op.text)

        # Process texts in batches
        batches = np.array_split(texts, len(texts) // op.batch_size + 1)
        log.info("Processing texts in batches", num_batches=len(batches))

        results = []
        for batch in tqdm(batches, desc="Ollama layer activations", smoothing=0.001):
            response = httpx.post(
                f"{self.server_url}/api/embed",
                json={
                    "model": model_name,
                    "input": [str(text) for text in batch],
                },
                timeout=60.0,
            )
            response.raise_for_status()
            arr = np.array(response.json()["embeddings"])
            results.append(arr)
            log.debug(
                "Processed batch", batch_size=len(batch), response_shape=arr.shape
            )

        log.debug("Concatenate")
        result = np.concatenate(results, axis=0)
        log.info("All batches processed", embedding_shape=result.shape)
        runner.write_numpy(op, result)


@register_model_provider("transformerlens", "tl")
class TransformerLensProvider(ModelProvider):
    """Provider for TransformerLens models with full layer and token support."""

    def get_layer_activations(self, runner, op: LLMLayerActivationsOp) -> np.ndarray:
        """Generate embeddings using TransformerLens."""
        log = logger.bind(model_name=op.model_name, op=op.uuid)
        if op.dtype is None:
            raise ValueError(
                "TransformerLens requires dtype to be specified. Suggest float32."
            )
        if op.max_length is None:
            raise ValueError("TransformerLens requires max_length to be specified.")
        if op.layer_num is None:
            raise ValueError("TransformerLens requires layer_num to be specified.")
        if op.token_mode not in ("last", "mean"):
            raise ValueError(
                "TransformerLens requires token_mode to be one of 'last', 'mean'."
            )
        import torch
        from transformer_lens import HookedTransformer, utils

        if op.torch_compile:
            raise ValueError(
                "TransformerLens does not support torch_compile=True. Use torch_compile=False."
            )

        # Materialize the text data
        texts = runner.to_numpy(op.text)

        device = self._detect_device(op.device)
        log = log.bind(device=device)

        if device == "mps":
            import torch
            from packaging import version

            if version.parse(torch.__version__) >= version.parse("2.8.0"):
                raise ValueError(
                    'TransformerLens\' MPS support is broken on torch >=2.8.0. Downgrade to torch 2.7 on your mac or use device="cpu". (Activations differ by ~0.5 between mps and cpu)'
                )

        # Load model
        _, model_name = get_model_provider(op.model_name)
        log.info("Loading TransformerLens model")
        model = HookedTransformer.from_pretrained_no_processing(
            model_name,
            device=device,
            dtype=op.dtype,
        )
        model.eval()

        # Handle negative layer indexing
        n_layers = model.cfg.n_layers
        layer_num = op.layer_num
        if layer_num < 0:
            layer_num = n_layers + layer_num
        if layer_num >= n_layers or layer_num < 0:
            raise ValueError(
                f"layer_num {layer_num} out of range for model with {n_layers} layers"
            )

        # Process in batches
        batches = [
            texts[i : i + op.batch_size] for i in range(0, len(texts), op.batch_size)
        ]
        results = []
        log = log.bind(num_batches=len(batches))

        torch.set_grad_enabled(False)

        for batch_idx, batch in enumerate(
            tqdm(batches, desc="TransformerLens layer activations", smoothing=0.001)
        ):
            # Apply chat template to batch
            input_batch = [[{"role": "user", "content": text}] for text in batch]
            input_tok = model.tokenizer.apply_chat_template(
                conversation=input_batch,
                padding=True,
                return_tensors="pt",
                add_generation_prompt=True,
                padding_side="right",
                truncation=True,
                max_length=op.max_length,
            ).to(device)

            blog = log.bind(batch_idx=batch_idx, batch_size=len(batch))

            # Get attention mask
            input_mask = utils.get_attention_mask(
                model.tokenizer,
                input_tok,
                model.cfg.default_prepend_bos,
            ).to(device)

            # Run model with cache
            if layer_num < 0:
                raise ValueError(f"layer_num must be >= 0, got {layer_num}")
            layer_key = f"blocks.{layer_num}.hook_resid_pre"
            _, activation_cache = model.run_with_cache(
                input_tok,
                names_filter=lambda name: name == layer_key,  # noqa: B023
            )

            # Extract activations for the specific layer
            layer_activations = (
                activation_cache[layer_key].detach().cpu().numpy().astype(op.dtype)
            )
            input_mask_cpu = input_mask.detach().cpu().numpy()
            blog.debug(
                "forward pass done",
                input_tok=input_tok,
                input_mask=input_mask,
                keys=list(activation_cache.keys()),
                activations_shape=layer_activations.shape,
            )

            # Process each sample in batch
            batch_results = []
            for i in range(len(batch)):
                if op.token_mode == "last":
                    # Last token
                    last_input_token = int(input_mask_cpu[i].sum()) - 1
                    embedding = layer_activations[i, last_input_token].copy()
                    # copy allows the rest of the tokens to be GC'd
                elif op.token_mode == "mean":
                    # Masked mean
                    mask = input_mask_cpu[i, ..., np.newaxis]
                    numel = mask.sum()
                    embedding = (
                        (layer_activations[i] * mask).sum(axis=0) / numel
                    ).astype(op.dtype)
                elif op.token_mode == "all":
                    raise NotImplementedError()
                    # All tokens as 2D array (n_tokens, embed_dim)
                    numel = int(input_mask_cpu[i].sum())
                    embedding = layer_activations[i, :numel]  # Keep 2D shape
                else:
                    raise ValueError(f"Unsupported token_mode: {op.token_mode}")

                batch_results.append(embedding)

            results.extend(batch_results)

            # Clear GPU cache
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()

        # Regular case - all embeddings have same shape
        log.debug("all batches processed, concatenating", n_batches=len(batches))
        results = np.array(results)
        runner.write_numpy(op, results)


@register_model_provider("huggingface", "hf")
class HuggingFaceProvider(ModelProvider):
    """Provider for HuggingFace Transformers models."""

    def get_layer_activations(self, runner, op: LLMLayerActivationsOp) -> np.ndarray:
        """Generate embeddings using HuggingFace Transformers."""
        log = logger.bind(op=op.uuid)
        if op.max_length is None:
            raise ValueError("HuggingFace requires max_length to be specified.")
        if op.dtype is None:
            raise ValueError(
                "HuggingFace requires dtype to be specified. Suggest float32."
            )

        if op.torch_compile:
            raise ValueError(
                "HuggingFace does not support torch_compile=True. Use torch_compile=False."
            )

        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Materialize the text data
        texts = runner.to_numpy(op.text)

        device = self._detect_device(op.device)
        log = log.bind(device=device)

        # Load model and tokenizer
        _, model_name = get_model_provider(op.model_name)
        log = log.bind(model_name=model_name)
        log.info("loading HuggingFace model")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=op.dtype,
            # device_map={"": device},
        )
        model.to(device)
        # note: there is a difference between from_pretrained(torch_dtype='float16') and model.half()
        model.eval()

        # Set padding token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        torch.set_grad_enabled(False)

        # Process texts in batches
        batches = [
            texts[i : i + op.batch_size] for i in range(0, len(texts), op.batch_size)
        ]
        results = []
        log = log.bind(num_batches=len(batches))

        for batch in tqdm(
            batches, desc="HuggingFace layer activations", smoothing=0.001
        ):
            # Apply chat template to batch
            inputs = tokenizer.apply_chat_template(
                [[{"role": "user", "content": str(text)}] for text in batch],
                add_generation_prompt=True,
                return_tensors="pt",
                return_attention_mask=True,
                return_dict=True,
                truncation=True,
                padding_side="right",
                max_length=op.max_length,
                padding=True,
            ).to(device)
            blog = log.bind(batch_size=len(batch), input=inputs)

            # Forward pass
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            # Extract hidden states
            hidden_states = outputs.hidden_states

            # Handle negative layer indexing
            if op.layer_num >= len(hidden_states):
                raise ValueError(
                    f"layer_num {op.layer_num} out of range for model with {len(hidden_states)} layers"
                )
            selected_layer = hidden_states[op.layer_num]
            blog.debug("forward pass done", activation_shape=selected_layer.shape)

            # Process each sample in batch
            batch_results = []
            for i in range(len(batch)):
                # Extract tokens based on mode
                if op.token_mode == "last":
                    # Last token
                    last_input_token = inputs["attention_mask"][i].sum().item()
                    embedding = selected_layer[i, last_input_token - 1, :].cpu().numpy()
                elif op.token_mode == "mean":
                    # Mean of all tokens (with attention mask)
                    attention_mask = inputs["attention_mask"][i].cpu().numpy()
                    mask = attention_mask[..., np.newaxis]
                    numel = mask.sum()
                    embedding = (selected_layer[i].cpu().numpy() * mask).sum(
                        axis=0
                    ) / numel
                else:
                    raise ValueError(
                        f"Unsupported token_mode for HuggingFace: {op.token_mode}. Supported: 'last', 'mean'"
                    )

                batch_results.append(embedding)

            results.extend(batch_results)

            # Clear GPU cache
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()

        log.debug("all batches processed, concatenating", n_batches=len(results))
        results = np.array(results)
        runner.write_numpy(op, results)


@register_model_provider("sentencetransformer", "st")
class SentenceTransformerProvider(ModelProvider):
    """Provider for HuggingFace Transformers models."""

    def get_layer_activations(self, runner, op: LLMLayerActivationsOp) -> np.ndarray:
        """Generate embeddings using HuggingFace Transformers."""
        log = logger.bind(op=op.uuid)
        if op.max_length is None:
            raise ValueError("SentenceTransformer requires max_length to be specified.")
        if op.dtype is None:
            raise ValueError(
                "SentenceTransformer requires dtype to be specified. Suggest float32."
            )
        if op.layer_num != -1:
            raise ValueError("SentenceTransformer requires layer_num to be -1.")

        if op.torch_compile:
            raise ValueError(
                "SentenceTransformer does not support torch_compile=True. Use torch_compile=False."
            )

        import numpy as np
        import torch
        from sentence_transformers import SentenceTransformer

        # Materialize the text data
        texts = runner.to_numpy(op.text)

        # Load model and tokenizer
        _, model_name = get_model_provider(op.model_name)
        log = log.bind(model_name=model_name)
        log.info("loading SentenceTransformer model")
        model = SentenceTransformer(model_name)

        torch.set_grad_enabled(False)

        # Process texts in batches
        log.info("Processing SentenceTransformer texts", n_texts=len(texts))
        encodings = model.encode(
            texts,
            show_progress_bar=True,
            batch_size=op.batch_size,
        )
        log.info("All batches processed", shape=encodings.shape)
        results = np.array(encodings)
        runner.write_numpy(op, results)
