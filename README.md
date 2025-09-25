# Krnel-graph

A lightweight Python library for building **strongly typed content-addressable computation graphs**, especially for mechanistic interpretability research.

Think of Krnel-graph as **"git for ML data transformations"** - every operation has a content hash of its parameters and dependencies. Results are cached and you can reproduce any computation exactly. Because the graph is strongly typed, **all operations are serializable and easily discoverable**, by you, your editor, and the agents you use.

Krnel-graph separates specification from implementation. Each operation's definition contains everything needed to materialize that operation, and each Runner can implement each operation differently. This lets you **swap in different backends**, dataflow executors, orchestrators, etc.

    TODO pretty figure, showing:
    - a nice graph, including custom ops
    - a HuggingfaceRunner() underneath
    - a NVidiaNemoRunner()
    - notebook w/ experiment results

We've tested krnel-graph on the following platforms:

- MacOS (arm64, MPS)
- Linux (amd64, CUDA)
- Windows native (amd64, CUDA)
- Windows WSL2 (amd64, CUDA)

## Quick start

Installation from PyPI with uv:

```bash
$ uv add krnel-graph[cli,ml]

# (Optional) Configure where Runner() saves results
# Defaults to /tmp
$ uv run krnel-graph config --store-uri /tmp/krnel/
# s3://, gs://, or any fsspec url supported
```

Make `main.py` with the following definitions:

```python
from krnel.graph import Runner
runner = Runner()

# Load data
ds_train   = runner.from_parquet('data_train.parquet')
col_prompt = ds_train.col_text("prompt")
col_label  = ds_train.col_categorical("label")

# Get activations from a small model
X_train = col_prompt.llm_layer_activations(
    model="hf:gpt2",
    layer=-1,
)

# Train a probe on contrastive examples
train_positives = col_label.is_in({"positive_label_1", "positive_label_2"})
train_negatives = ~train_positives
probe = X_train.train_classifier(
    positives=train_positives,
    negatives=train_negatives,
)

# Get test activations by substituting training set with testing set
# (no need to repeat the entire graph)
ds_test = runner.from_parquet('data_test.parquet')
X_test = X_train.subs((ds_train, ds_test))

test_scores = probe.predict(X_test)
eval_result = test_scores.evaluate(
    gt_positives=train_positives.subs((ds_train, ds_test)),
    gt_negatives=train_negatives.subs((ds_train, ds_test)),
)

if __name__=="__main__":
    # All operations are lazily evaluated until materialized:
    print(runner.to_json(eval_result))
```

Then, inspect the results in a notebook:

```python
from main import runner, eval_result, X_train

# Materialize everything and print result:
print(runner.to_json(eval_result))

# Display activations of training set (GPU-intense operation)
print(runner.to_numpy(X_train))
```

Or use the (completely optional) `krnel-graph` CLI to materialize a selection of operations and/or monitor progress:

```shell
# Run parts of the graph
$ uv run krnel-graph run -f main.py -t LLMLayerActivations   # By operation type
$ uv run krnel-graph run -f main.py -s X_train               # By Python variable name

# Show status
$ uv run krnel-graph summary -f main.py

# Diff the pseudocode of two graph operations
$ uv run krnel-graph print -f main.py -s X_train > /tmp/train.txt
$ uv run krnel-graph print -f main.py -s X_test > /tmp/test.txt
$ git diff --no-index /tmp/train.txt /tmp/test.txt
```

## What this library is

Krnel-graph is a content-addressable dataflow library that provides:

1. ✅ **An extensible palette of mechanistic interpretability operations** for training, running, and evaluating linear probes on existing datasets in batch...
    - *Excelent editor support* via autocomplete, type hints, docstrings, etc
2. ✅ ...alongside **a reference implementation of these operations**, with optional integrations to Huggingface, TransformerLens, Ollama, and other inference fabric...
3. ✅ ...all built on top of **a lightweight computation graph flow library**, featuring:
    - *Built-in model and data provenance* via automatic dependency tracking
    - *Cached, reproducible results* through content-addressable operations
    - *Immutable operation specifications* with deterministic UUIDs
    - *Fluent API* for building complex data pipelines
    - *ML-first design* with built-in support for embeddings, classifiers, and LLMs
    - (Optional) *Local execution* with Arrow/Parquet storage (filesystem / GCS / S3 / ...)

## What this library is not

- ❌ ...a **task orchestrator** like Airflow or Prefect
    - No YAML templates, no Docker containers (by default)
- ❌ ...a **distributed computing framework** like Dask or Ray
    - The default runner uses local-only execution for now
    - Results can be saved and loaded to a remote store (NFS, GCS/S3, ...)
    - Bring your own scheduling / workflow management if needed
- ❌ ...an **experimentation or visualization tool** (though it integrates nicely with notebooks and plotting libraries)

The goal of krnel-graph is to separate well-typed specifications from their implementation. Krnel-graph does not depend on particular infrastructure. All operations are separated from their implementations, so it's easy to swap in your own dataflow executor if you prefer.

## Core Concepts

### OpSpec: Content-Addressable Operations

Every operation in Krnel is an `OpSpec` - an immutable specification with a deterministic UUID:

```python
from krnel.graph import LoadInlineJsonDatasetOp

# These two operations have identical UUIDs
op1 = LoadInlineJsonDatasetOp(data={'x': [1, 2, 3]})
op2 = LoadInlineJsonDatasetOp(data={'x': [1, 2, 3]})
assert op1.uuid == op2.uuid
```

Krnel uses a type-driven fluent API where each column type provides relevant methods:

```python
dataset = LoadInlineJsonDatasetOp(data={
    'text': ['Hello', 'World'],
    'embeddings': [[0.1, 0.2], [0.3, 0.4]],
    'labels': ['A', 'B']
})

# Type-specific operations
text_col = dataset.col_text('text')           # TextColumnType
vector_col = dataset.col_vector('embeddings')  # VectorColumnType
category_col = dataset.col_categorical('labels') # CategoricalColumnType

# Chaining operations
generated_text = vector_col.train_classifier(...).predict(...some_other_vector_col ...)
```

### Runners: Execution Engines

Runners execute your computation graphs.

```python
from krnel.graph.runners.local_runner import LocalArrowRunner

# This runner saves into local memory:
runner = LocalArrowRunner(store_uri="memory://")

# Different output formats
arrow_table = runner.to_arrow(my_operation)
numpy_array = runner.to_numpy(my_operation)
json_data = runner.to_json(my_operation)

# The default runner can be configured via `krnel-graph config`
from krnel.graph import Runner
runner = Runner()
```

## Writing custom operations

### 1. Define your operation class

```python
from krnel.graph import OpSpec
from krnel.graph.types import TextColumnType, VectorColumnType

class MyCustomEmbeddingOp(VectorColumnType):
    """Extract embeddings using a custom model."""
    text_input: TextColumnType
    model_path: str
    max_length: int = 512
```

### 2. Implement the execution logic

```python
from krnel.graph.runners.local_runner import LocalArrowRunner
import pyarrow as pa

# Dispatch happens by type annotation:
@LocalArrowRunner.implementation
def my_custom_embedding_impl(runner, op: MyCustomEmbeddingOp):
    """Implementation that gets called when this op is executed."""

    # Get input data
    text_data = runner.to_arrow(op.text_input)
    texts = text_data.column(0).to_pylist()

    # Your custom logic here
    embeddings = []
    for text in texts:
        # Load your model, extract embeddings, etc.
        embedding = extract_embedding(text, op.model_path, op.max_length)
        embeddings.append(embedding)

    runner.write_arrow(op, pa.array(embeddings))

def extract_embedding(text: str, model_path: str, max_length: int):
    # Your embedding extraction logic
    return [0.1, 0.2, 0.3]  # placeholder
    ...
```

### 3. Use your custom operation

```python
dataset = LoadInlineJsonDatasetOp(data={'text': ['Hello world', 'Custom ops!']})
text_col = dataset.col_text('text')

# Using your custom operation
embeddings = text_col.my_custom_embedding(
    model_path='./my-model',
    max_length=256
)

result = runner.to_numpy(embeddings)
```
