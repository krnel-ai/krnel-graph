# Krnel-graph
### [Blog](https://krnel.ai/blog/2025-10-21-krnel-graph/) • [Docs](https://krnel-graph.readthedocs.io/en/latest/) • [Examples](https://github.com/krnel-ai/krnel-graph/tree/main/examples) • [Github](https://github.com/krnel-ai/krnel-graph) • [PyPI](https://pypi.org/project/krnel-graph/)

A **Python toolbox for mechanistic interpretability research** built on a **lightweight strongly-typed computation graph spec.**

- **Run language models** using HuggingFace Transformers, TransformerLens, Ollama, *etc.,* and save activations from the residual stream
- **Train linear probes** from cached activations and evaluate their results
- **Fetch logit scores** for guardrail models
- Load and prepare datasets

### Applications

- **Build better guardrails** using linear probes that understand model internals
- **Explore large datasets** grouped by semantic similarity
- **Vizualize high-dimensional embeddings** with built-in UMAP scatterplots
- Evaluate derivative experiments quickly with **full caching and provenance tracking** of results.
- **Infrastructure-agnostic**: Run in a notebook, on your GPU machine's CLI, or via the task orchestration framework of your choice!

![Krnel-graph figure](https://raw.githubusercontent.com/krnel-ai/krnel-graph/main/docs/_static/krnel-graph-hero.webp)

## Quick start

Krnel-graph works on the following platforms:

- MacOS (arm64, MPS, Apple M1 or better)
- Linux (amd64, CUDA)
- Windows native (amd64, CUDA)
- Windows WSL2 (amd64, CUDA)

Install from PyPI with uv:

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

