# Krnel-graph

A lightweight Python library for building **strongly typed content-addressable computation graphs**, especially for mechanistic interpretability research.

Krnel-graph lets you define data transformations and ML workflows as immutable, cacheable operations that automatically track their dependencies and results.
- Use our bulit-in operations to train, run, and evaluate linear probes on your own datasets
- (Optionally) add your own codebase-specific operations

Think of Krnel-graph as **"git for ML data transformations"** - every operation has a content hash, results are cached, and you can reproduce any computation exactly.

Krnel-graph is implementation-agnostic. We have default implementations for operations on HuggingFace Transformers, but you can swap out your own, or even use your own dataflow library!

## Quick start

Installation from PyPI:

```bash
$ uv add krnel-graph[cli]

# Configure where Runner() saves results
# (s3://, gs://, or any fsspec url supported)
$ uv run krnel-graph config --store-uri /tmp/krnel/
# Defaults to /tmp
```

Define a graph of operations to run in a file, like `main.py`:

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

Or use the `krnel-graph` CLI to materialize a selection of operations and monitor progress:

```shell
# Run parts of the graph
$ krnel-graph run -f main.py -t LLMLayerActivations   # By operation type
$ krnel-graph run -f main.py -n X_train               # By Python variable name

# Show status
$ krnel-graph summary -f main.py

# Diff the pseudocode of two graph operations to see what the difference is
$ krnel-graph print -f main.py -n X_train > /tmp/train.txt
$ krnel-graph print -f main.py -n X_test > /tmp/test.txt
$ git diff --no-index /tmp/train.txt /tmp/test.txt
```

## What this library is

Krnel-graph is a content-addressable dataflow library that provides:

1. ✅ **An extensible selection of mechanistic interpretability operations** for training, running, and evaluating linear probes on existing datasets in batch...
2. ✅ ...alongside **a reference implementation of these operations**, with optional integrations to Huggingface, TransformerLens, Ollama, and other inference fabric...
3. ✅ ...all built on top of **a lightweight computation flow graph library**, featuring:
    - *Built-in model and data provenance* via automatic dependency tracking
    - *Cached, reproducible results* through content-addressable operations
    - *Immutable operation specifications* with deterministic UUIDs
    - *Fluent API* for building complex data pipelines
    - *ML-first design* with built-in support for embeddings, classifiers, and LLMs
    - *Local execution* with Arrow/Parquet storage (local filesystem or GCS / S3 / ...)

## What this library is not

Krnel-graph is meant to be lightweight. It is not:

- ❌ ...a **task orchestrator** like Airflow or Prefect (no scheduling/workflow management)
- ❌ ...a **distributed computing framework** like Dask or Ray (local execution only for now)
- ❌ ...a **database** or data warehouse (though it caches intermediate results)
- ❌ ...a **visualization tool** (though it integrates with plotting libraries)

- **Krnel-graph is not implementation-specific.** All operations are separated from their implementations, so it's easy to swap in your own dataflow executor if you prefer.


## Core Concepts

### OpSpec: Content-Addressable Operations

Every operation in Krnel is an `OpSpec` - an immutable specification with a deterministic UUID:

```python
from krnel.graph import FromListOp

# These two operations have identical UUIDs
op1 = FromListOp(data={'x': [1, 2, 3]})
op2 = FromListOp(data={'x': [1, 2, 3]})
assert op1.uuid == op2.uuid
```

Krnel uses a type-driven fluent API where each column type provides relevant methods:

```python
dataset = FromListOp(data={
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
dataset = FromListOp(data={'text': ['Hello world', 'Custom ops!']})
text_col = dataset.col_text('text')

# Using your custom operation
embeddings = text_col.my_custom_embedding(
    model_path='./my-model',
    max_length=256
)

result = runner.to_numpy(embeddings)
```
