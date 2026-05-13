# Agent guide: extending krnel-graph

This file is for AI agents (and humans) working with the library — whether writing new operations, extending runners, or understanding the codebase from scratch. It covers both the high-level architecture and the patterns that aren't obvious from reading the type signatures.

## Architecture Overview

Krnel is a library for building content-addressable computation graphs. The architecture has three main layers:

### 1. Graph Layer (`src/krnel/graph/`)
- **OpSpec** (`op_spec.py`): Core abstraction representing immutable nodes in a computation DAG. Each OpSpec has a deterministic UUID based on its content and implements content-addressable semantics.
  - `dataset_ops.py`: Dataset loading, column selection, row operations
  - `classifier_ops.py`: Classification training operations
  - `llm_ops.py`: LLM-based operations and layer activations
  - `viz_ops.py`: Visualization operations (UMAP, etc.)
- **Operation Types** (`types.py`): The entire user-facing fluent API lives here. Client code should prefer these functions instead of instantiating `OpSpec`s manually.
- **Graph Transformations** (`graph_transformations.py`): Functions for DAG traversal, dependency analysis, and graph manipulation.

### 2. Runners Layer (`src/krnel/graph/runners/`)
- **LocalArrowRunner** (`local_runner.py`): Executes operations locally, caching results as Arrow Parquet files using fsspec for storage abstraction.
- **ModelProvider** (`model_registry.py`): Registry system for ML model providers and activation extraction.
- **Result Conversion Methods**: Each runner provides `to_numpy()`, `to_arrow()`, `to_json()` methods for accessing computation results in different formats.
- **OpStatus** (`op_status.py`): Tracks execution status of operations.

### 3. CLI and Utilities
- **CLI** (`cli.py`): Command-line interface built with cyclopts for operation status checking and materialization.
- **Visualization** (`viz/`): UMAP-based visualization utilities.

## Key Concepts

- **Content-Addressable Operations**: Every OpSpec has a deterministic UUID computed from its content. Identical operations producing the same data always produce the same UUID.
- **DAG Semantics**: OpSpec fields referencing other OpSpecs create DAG edges. Scalar fields are treated as parameters.
- **Immutability**: OpSpecs cannot be modified after creation. Changes create new OpSpecs.
- **ExcludeFromUUID**: Use `Annotated[Type, ExcludeFromUUID()]` to exclude fields from UUID computation while keeping them for provenance. See the dedicated section below.

## Build and Development Commands

This project uses `uv` for Python package management. Key commands:

- **Run tests**: `make test` or `uv run --all-extras pytest -v`
- **Run specific test**: `uv run --all-extras pytest tests/test_graph.py -v`
- **Run specific test function**: `uv run --all-extras pytest tests/test_graph.py::test_function_name -v`
- **Run tests with coverage**: `uv run --all-extras pytest -v --cov=src/krnel --cov-report=term --cov-report=xml`
- **Build documentation**: `uv run --extra docs sphinx-build -b html docs docs/_build/html`
- **Build package**: `uv build`

Use `make test` as a shortcut for running tests with all extras.

## Testing

Tests are in `tests/` using pytest. The project follows **Test-Driven Development (TDD)** — failing tests define expected behavior before implementation.

The test suite covers:
- OpSpec serialization/deserialization and UUID computation
- Graph transformation operations
- Runner execution and caching behavior
- LocalArrowRunner operations (`test_local_arrow_runner.py`)

### Testing Philosophy
- **Failing tests are specifications** — they define what needs to be implemented next.
- **No skipping or special-casing unimplemented features** — tests fail explicitly to show implementation gaps.

Key test files:
- `test_graph.py` — Core OpSpec functionality
- `test_graph_transformations.py` — DAG manipulation
- `test_local_arrow_runner.py` — LocalArrowRunner operations (LoadInlineJsonDatasetOp, SelectColumnOp, TakeRowsOp, etc.)

## Optional Dependencies

Optional dependency groups in `pyproject.toml`:
- `test`: Testing dependencies (pytest, pytest-cov)
- `viz`: Visualization dependencies (umap-learn, jupyter-scatter, seaborn, numba)
- `cli`: CLI dependencies (rich, cyclopts)
- `docs`: Documentation dependencies (sphinx, sphinx-autoapi, etc.)

Install with: `uv sync --extra <group_name>` or `--all-extras` for everything.

## Working conventions across Krnel projects

These apply to `krnel-graph` and to every research project that builds on it.

### GitHub authentication on a new machine

Every Krnel-owned `pyproject.toml` carries a leading comment block describing how to set up GitHub access without password prompts. The short version: run `gh auth login -p ssh -h github.com -w` once per machine; for throwaway GCP VMs, prefer `gcloud compute ssh <instance> -- -A` (SSH agent forwarding) so no key lands on the VM. See the comment at the top of any project's `pyproject.toml`.

---

## The type hierarchy

Every OpSpec has a return type baked into its class hierarchy. An op that produces a column of vectors inherits from `VectorColumnType`, not just `OpSpec`. This is how the fluent API works: the methods on `VectorColumnType` are available on anything that produces vectors.

```python
from krnel.graph import OpSpec, VectorColumnType, ExcludeFromUUID
from typing import Annotated

# The op inherits from its *output* type, not from OpSpec directly.
class MyEmbeddingOp(VectorColumnType):
    source: VectorColumnType          # OpSpec-typed field → DAG edge (dependency)
    scale_factor: float = 1.0         # scalar field → parameter (not a DAG edge)
    description: Annotated[str, ExcludeFromUUID()] = ""  # excluded from UUID
```

The full type hierarchy lives in `src/krnel/graph/types.py`. The concrete operations live in `*_ops.py` files. When adding a new op, inherit from the appropriate `*Type` class.

## Fields: dependencies vs. parameters

The rule is simple: **if a field is typed as an `OpSpec` subclass, it is a DAG edge**. Everything else is a parameter.

```python
class MyOp(ScoreColumnType):
    input: VectorColumnType      # dependency — creates a DAG edge to whatever produced the vectors
    threshold: float = 0.5       # parameter — stored in UUID computation, not a graph node
```

DAG edges are traversed by runners and by `subs()`. Parameters are not.

## ExcludeFromUUID

Fields annotated with `ExcludeFromUUID()` are serialized for provenance but excluded from UUID computation. Use this for fields that don't affect results: local file paths, human-readable labels, debug flags.

```python
from typing import Annotated
from krnel.graph import ExcludeFromUUID

class MyOp(DatasetType):
    data: DatasetType
    label: Annotated[str, ExcludeFromUUID()] = ""  # won't change the UUID
```

Two `MyOp` instances with the same `data` but different `label` values will have the same UUID and share cached results.

Because ML frameworks are broadly nondeterministic, it's usually best to keep batch size, device, and similar parameters *inside* UUID computation rather than excluding them — silent fallbacks make the cache lie about what was computed.

## EphemeralOpMixin

Inherit from `EphemeralOpMixin` when an op is cheap enough to always re-run rather than cache. The runner skips writing results to storage for ephemeral ops.

```python
from krnel.graph.op_spec import EphemeralOpMixin

class SliceRowsOp(DatasetType, EphemeralOpMixin):
    dataset: DatasetType
    start: int = 0
    end: int | None = None
```

Examples: `TakeRowsOp`, `SelectColumnOp`, `BooleanLogicOp`. Use it for any operation that is a pure in-memory transformation with negligible cost.

## Adding a runner implementation

After defining the OpSpec, register an implementation on the runner. The decorator infers the op type from the second parameter's annotation.

```python
import pyarrow as pa
from krnel.graph.runners.local_runner import LocalArrowRunner
from my_module import MyEmbeddingOp

@LocalArrowRunner.implementation
def run_my_embedding_op(runner: LocalArrowRunner, op: MyEmbeddingOp) -> None:
    source_table = runner.to_arrow(op.source)        # materialize dependency
    # ... compute result ...
    result = pa.array([...])
    runner.write_arrow(op, result)                   # write result to storage
```

The implementation must call one of `runner.write_arrow()`, `runner.write_json()`, `runner.write_numpy()`, or `runner.write_sklearn_estimator()` to persist the result. The return value is ignored.

Implementations are registered globally when the module is imported, so make sure the module containing `@LocalArrowRunner.implementation` is imported before the runner is used.

## subs(): swapping graph nodes

`subs()` reconstructs the graph with targeted changes. The most common usage is swapping a dataset to re-run a subgraph on new data:

```python
# Build the graph once
ds_train = runner.from_parquet("train.parquet")
X = ds_train.col_text("text").llm_layer_activations(model="hf:gpt2", layer=-1)

# Reuse the graph with a different dataset — no re-specification needed
ds_test = runner.from_parquet("test.parquet")
X_test = X.subs(ds_train, file_path="test.parquet")  # update a field on ds_train
# or equivalently:
X_test = X.subs((ds_train, ds_test))                  # swap the node wholesale
```

`subs()` does not mutate anything. It always returns a new OpSpec.

## Working with Runner

`Runner` is a factory function, not a class. Calling `Runner()` returns a `LocalArrowRunner` by default (or whatever is configured). Subsequent calls with the same arguments return the cached instance.

```python
from krnel.graph import Runner
runner = Runner()                  # returns LocalArrowRunner by default
result = runner.to_json(some_op)   # dict
table  = runner.to_arrow(some_op)  # pa.Table
array  = runner.to_numpy(some_op)  # np.ndarray
df     = runner.to_pandas(some_op) # pd.DataFrame
```

Every op also has a `.runner` property that auto-discovers the runner from its dependency chain. So once any node in the graph is rooted in a `runner.from_parquet(...)` (or similar), you can call `.to_numpy()` directly on a derived op without passing the runner again:

```python
ds = runner.from_parquet("data.parquet")   # op._runner is set
X = ds.col_text("prompt").llm_layer_activations(...)
X.to_numpy()                                # finds runner via dependency walk
```

Ops loaded from disk via `runner.uuid_to_op(uuid)` have `_runner` populated recursively, so the fluent API keeps working on them.

See `examples/02-custom-op/` for an end-to-end example of a custom op with a runner implementation.

## Storage method reference

The runner's `write_*` and `to_*` methods are paired by storage format. Pick the one that matches what your implementation produces:

| Op output                        | Implementation writes with              | Consumers read with                | On-disk file       |
|----------------------------------|-----------------------------------------|------------------------------------|--------------------|
| Arrow table / array (most ops)   | `runner.write_arrow(op, t)`             | `to_arrow`/`to_numpy`/`to_pandas`  | `result.parquet`   |
| Numpy array (1D scalar, 2D vec)  | `runner.write_numpy(op, a)`             | same as above                      | `result.parquet`   |
| JSON-serializable dict (reports) | `runner.write_json(op, d)`              | `to_json`                          | `result.json`      |
| Trained sklearn estimator        | `runner.write_sklearn_estimator(op, m)` | `to_sklearn_estimator`             | `result.pickle`    |

`write_numpy` is sugar for `write_arrow` — 1D arrays become single scalar columns; 2D arrays become a single `FixedSizeList` column (preserving vector shape). If you write a 2D numpy array, `to_numpy()` reshapes it back to 2D automatically on read.

JSON columns (`JSONColumnType` outputs) are still backed by Arrow in storage — see `jq_op.py` in [`research/roblox-guard`](https://github.com/krnel-ai/research/tree/main/roblox-guard) for the pattern: build a Python list of dicts/lists, `pa.array(...)` it, and call `write_arrow`. Reserve `write_json` for ops whose entire result is a single JSON document (e.g. `ClassifierEvaluationOp`).

Sometimes for conversational types (still in progress), it's better to store long JSON objects as serialized text. For an example of this pattern, see the OpenAI conversation parsing machinery in [`krnel-blanket-base/src/krnel/blanket/adapters`](https://github.com/krnel-ai/krnel-blanket-base/tree/main/src/krnel/blanket/adapters)

## Container fields as DAG edges

The "OpSpec-typed field = DAG edge" rule extends through container types. The framework walks `Annotated`, `Union`, `list`, `tuple`, `set`, and `dict` annotations looking for OpSpec subclasses:

```python
class JinjaTemplatizeOp(TextColumnType):
    template: str
    context: dict[str, TextColumnType | JSONColumnType]   # each value is a DAG edge
    constants: dict[str, float | int | str]                # no OpSpecs → parameters
```

Each value in `context` becomes a real graph dependency: it's traversed by `get_dependencies(recursive=True)`, considered by `subs()`, and uploaded by `runner.prepare()`. Dict *keys* are always parameter data, never edges. See `jq_op.py` and `JinjaTemplatizeOp` for working examples.

Practical implications:
- Don't store `OpSpec` instances inside `list`/`dict` parameters if you don't want them traversed — they will be.
- Conversely, if you want a value to be a dependency, type-annotate it as an OpSpec subclass even when nested.
- The container shape is stable in the UUID — Python dicts are insertion-ordered, so building `context` dicts in different orders changes the hash. Sort keys when building dynamically if you want stability.

## Implementation dispatch and inheritance

`@LocalArrowRunner.implementation` registers the function against the *exact* class in the second parameter's annotation. At dispatch time, the runner walks the op's MRO to find a matching implementation. Consequences:

- **Subclasses inherit implementations from their parent class.** If `MyCustomActivationsOp(LLMLayerActivationsOp)` doesn't register its own implementation, the parent's runs.
- **Multiple matching implementations is an error.** Don't register two implementations against the same class or against `OpSpec` directly.
- **Registration happens at import time.** The implementation module must be imported before the op is materialized; otherwise the runner reports "No implementation for X." A common idiom is to define the op and its `@LocalArrowRunner.implementation` in the same module, so importing the op also registers the implementation.

## Importing op classes before deserialization

`graph_deserialize()` (used by `runner.uuid_to_op()` and `krnel-graph run -u UUID`) reconstructs ops by walking `OpSpec.__subclasses__()` to find a class matching the stored `"type"` string. **The class must be imported in the current Python process** or you'll get `Class with name 'XYZ' not found in OpSpec hierarchy`.

In notebooks/scripts that consume ops produced by another project, import every custom op module up front even if you don't reference it directly. For CLI invocations on a fresh machine (typically a GPU VM), the right move is to launch through a project-specific `run-krnel-graph` entrypoint that imports its own modules — see the section above. Falling back to `uv run krnel-graph run -u ...` from inside the project folder also works, since `uv run` activates the project's environment.

`find_subclass_of` also rejects ambiguous matches (two classes with the same name in different modules). The error suggests restarting your kernel; the real fix is usually deduplicating the modules — e.g. don't have both `research/foo/encoding_ops.py` and `research/bar/encoding_ops.py` importable at the same time with classes of the same name.

## Model providers (extending `llm_layer_activations` / `llm_logit_scores`)

`LLMLayerActivationsOp` and `LLMLogitScoresOp` route by URL scheme rather than by writing a new op. The model name is parsed as `scheme:model_name`, and the scheme is looked up in the model provider registry:

```python
from krnel.graph.runners.model_registry import (
    ModelProvider, register_model_provider, get_model_provider,
)
from krnel.graph.llm_ops import LLMLayerActivationsOp, LLMLogitScoresOp

@register_model_provider("vllm")                # also accepts aliases: ("foo", "f")
class VLLMProvider(ModelProvider):
    def get_layer_activations(self, runner, op: LLMLayerActivationsOp):
        _, model_name = get_model_provider(op.model_name)   # strips the "vllm:" prefix
        # ... compute activations, call runner.write_numpy(op, ...)
    def get_llm_output_logits(self, runner, op: LLMLogitScoresOp):
        ...
```

Built-in schemes: `hf:` (HuggingFace transformers), `tl:` (TransformerLens), `ollama:`. See `local_runner/model_registry_implementations.py`.

**Extending an existing provider** is a powerful pattern when you only need to change how the model is loaded. See (`research/roblox-guard/peft_model_provider.py`)[https://github.com/krnel-ai/research/blob/main/roblox-guard/peft_model_provider.py]:

```python
@register_model_provider("peft2")
class PeftModelProvider(HuggingFaceProvider):
    def _load_model(self, op):
        scheme, adapter, base = op.model_name.split(":")   # "peft2:adapter:base"
        ...
        return peft.PeftModel.from_pretrained(model, adapter), tokenizer
```

The base `HuggingFaceProvider` handles all the batching, chat templating, token-mode aggregation, and device management — the subclass only customizes `_load_model`. Schemes can encode multiple colon-separated arguments because `partition(":")` only splits on the first colon (so `peft2:meta-llama/Llama-3.1-8B:my-adapter` parses cleanly into a scheme and a model name that contains a colon).

Provider implementations should read assumptions out of the op (`op.dtype`, `op.max_length`, `op.token_mode`, etc.) and raise loudly when a field isn't supported — *don't silently ignore an unsupported parameter*. The UUID encodes the parameter, so silent fallbacks make the cache lie about what was computed.

## Classifier model registry

`VectorColumnType.train_classifier(model_type="...", ...)` looks up `model_type` in a separate registry of scikit-learn-compatible factories. To add a new probe:

```python
from krnel.graph.runners.local_runner.probe_implementations import register_classifier_model

@register_classifier_model("lr_nystroem")
def _create_lr_nystroem(params):
    import sklearn.pipeline, sklearn.kernel_approximation, sklearn.linear_model
    return sklearn.pipeline.make_pipeline(
        sklearn.kernel_approximation.Nystroem(**params.get("nystroem_params", {})),
        sklearn.linear_model.LogisticRegression(**params.get("lr_params", {})),
    )
```

The factory receives the dict that the caller passed as `params=...` to `train_classifier`. Whatever it returns must implement `fit(X, y)` plus one of `decision_function` or `predict_proba` (the predict implementation prefers `predict_proba` for binary problems). Pipelines work; calibrated classifiers work; anything sklearn-shaped works.

The trained model is serialized with `pickle` via `write_sklearn_estimator`, so it must be picklable — no closures over local state, no lambdas as estimator params.

## Sharing datasets across projects via UUID

`runner.uuid_to_op(uuid)` hydrates an op from its stored status JSON in the bucket, returning a fully-typed OpSpec with `_runner` populated. This is how research projects reuse each other's datasets and intermediate results without redefining them:

```python
guardbench_ds = runner.uuid_to_op(
    "LoadLocalParquetDatasetOp_70eb97af0cf056f56a72fa07cef959e31b0c8ba5f1548e5c7b45aa406d729302"
)
train_mask = guardbench_ds.col_train_test_split("split").train
```

The op behaves identically to one you constructed locally — `subs()`, `train_classifier()`, `predict()`, etc. all work. Pattern: the producing project prints a dataset op's UUID from its `main.py`; downstream projects hard-code the UUID and call `uuid_to_op` to bring it in. Hashes change if dataset bytes change, so a hard-coded UUID is a content-addressed reference, not a pointer that can drift.

Caveat: the producing op's *class* must still be importable (see "Importing op classes before deserialization") and the producing run must have written a status JSON (which `prepare()` and normal materialization do automatically).

## `runner.prepare(op)`

`prepare()` walks the graph and uploads any `LoadLocalParquetDatasetOp` whose bytes aren't already in the backing store. It does *not* run any compute. The intended flow for GPU jobs is:

1. On your laptop, build the graph and call `runner.prepare(op)` — this hashes local parquet files and pushes them to GCS.
2. Print `op.uuid`, SSH to the GPU box.
3. On the GPU box, run `uvx krnel-graph[ml] run -u <UUID>`. The runner hydrates from storage, finds the (now uploaded) parquet, and computes.

Without `prepare()`, the remote runner has the spec but not the bytes and will fail when it tries to materialize the dataset.

## `GroupedOp`: batching multiple ops

`GroupedOp(ops=[op1, op2, op3, ...])` is a thin wrapper that materializes each child in sequence and returns the last result. Use it when you want a single root node to drive a sweep — e.g. grid search across hyperparameter combinations — so `krnel-graph status` or `runner.prepare()` see one entry instead of N:

```python
all_ops = kg.GroupedOp(ops=[
    results.subs(clsf, model_type=m, params=p) for (m, p) in CLSF_PARAMS
])
```

See `research/guardbench-test/main.py` for the full grid-search pattern.

## `subs()` deep-target substitution

`subs()` accepts a target node anywhere in the graph, not just direct dependencies. The common idiom for swapping out the trained model inside a long evaluation chain:

```python
# results_guardbench = X.train_classifier(...).predict(X).evaluate(...)
# Want to retrain on a different dataset but keep the evaluation logic identical:
results_krnelharmful = results_guardbench.subs(
    results_guardbench.score.model,         # navigate to the TrainClassifierOp deep in the graph
    x=X.subs(text=kh_prompt, max_length=2048),
    positives=kh_label.is_in({...}),
    train_domain=kh_subset.train,
)
```

This works because `subs(target, **changes)` is equivalent to `subs([(target, target.subs(**changes))])` — the target is updated in isolation, then swapped in throughout the graph. You can chain substitutions or bundle them as a list of `(before, after)` pairs (see the `subs()` docstring in `op_spec.py`).

## CLI patterns

`main.py` should just build ops as top-level Python variables. The `krnel-graph` CLI (installed by the `cli` extra; invoke as `uv run krnel-graph` from a project, or `uvx krnel-graph[ml]` standalone) does the rest — don't reimplement materialization machinery inside `main.py` itself.

Subcommands: `status`, `summary`, `print`, `materialize` (alias `run`), `make-group`, `config`. Each takes the same filtering vocabulary, so a single graph definition powers all of them:

| Flag             | Picks ops that…                                                  |
|------------------|------------------------------------------------------------------|
| `-f main.py`     | are reachable from a top-level OpSpec/Runner variable in `main.py` |
| `-u <UUID>`      | match this UUID (substring OK). Can repeat. Works without `-f`.   |
| `-s <name>`      | are bound to a Python variable whose name matches (substring OK)  |
| `-t <Type>`      | are instances of this op class (substring OK)                     |
| `-p <value>`     | have any field equal to / containing this value                   |
| `-S <text>`      | match this string anywhere in their `to_code()` pseudocode        |
| `--state <s>`    | are in this runtime state (`new`/`running`/`completed`/`failed`/…) |
| `-n <N>`         | cap at N ops after all other filters                              |
| `--no-deps`      | don't pull in transitive dependencies (off by default)            |

Typical uses:

```sh
# What's the cache state of everything this project builds?
uv run krnel-graph summary -f main.py

# Run all activation-extraction ops (suitable for the GPU box)
uv run krnel-graph run -f main.py -t LLMLayerActivations

# Run just one named experiment from main.py
uv run krnel-graph run -f main.py -s eval_result

# Materialize one specific UUID, no main.py needed
uv run krnel-graph run -u TrainClassifierOp_d9dff236...

# Show the full graph pseudocode for any matching op
uv run krnel-graph print -f main.py -s probe
```

The `run` subcommand has built-in sharding for parallelism: `--shard-count N --shard-idx I` (combined with `--no-shuffle` if you want determinism). Don't build `show-evals | parallel` pipelines around `main.py` — let the CLI handle dispatching.

## Running graphs that use custom ops (`--import` / `--with`)

`uvx krnel-graph[ml] run -u <UUID>` only knows about ops defined in `krnel-graph` itself. If your graph involves ops from another package — `VLLMGenerateTextOp` from `krnel-blanket-base`, custom encoding ops from a research project, etc. — deserialization raises `Class with name 'XYZ' not found in OpSpec hierarchy` because those classes were never imported.

Solve this with two flags:

- `uvx --with git+<repo>` adds a package to `uvx`'s ephemeral environment so its modules can be imported.
- The `krnel-graph` CLI's `--import <module>` (alias `-m`, also spelled `--with`), repeatable, imports the named modules before doing any work. This is enough to register custom OpSpec subclasses and their runner implementations via side-effect imports.

Combined:

```sh
uvx --with git+https://github.com/krnel-ai/krnel-blanket-base.git krnel-graph \
    print -m krnel.blanket.adapters.vllm_hook \
    -u GroupedOp_7a686d8873fc2604d0905533229eb10ec1cc622c6482db7063874b1dee6f66c3
```

This works for every subcommand (`status`, `summary`, `print`, `run`/`materialize`, …) and gives you two properties for free:

1. **Pinned to a specific git revision.** `--with git+...` checks out the package at the requested ref, so you're running the exact source that produced the graph definition.
2. **Custom ops register before deserialization** because `-m` runs `importlib.import_module()` ahead of any UUID lookup.

This is the default runbook for "how do I run an op on the GPU box when the graph uses custom ops?" — prefer it over `uvx krnel-graph[ml] run -u`.

### Alternative: project-specific `run-krnel-graph` console script

If a project ships *many* op modules and you don't want callers to list each one with `-m`, expose a thin console script that imports them at the top and dispatches to the runner. `krnel-blanket-base` did this before `--import` existed; see `src/krnel/blanket/adapters/__main__.py` and its `[project.scripts]` entry. Trade-off: callers don't need to know which modules to import, but they also can't pick a subcommand other than the one the wrapper hardcodes. With `--import` now available, this is mostly historical — reach for it only if you find yourself repeating the same `-m` list a lot.

## Debugging utilities

A few introspection helpers that save real time:

- `op.to_code()` / `print(op)` — renders the full subgraph as Python pseudocode. Best way to understand what a hydrated op actually contains.
- `op.diff(other_op)` — structured graph diff; renders nicely in notebooks. Useful for "why don't these two UUIDs match?"
- `op._repr_html_` (just display in Jupyter) — Mermaid flowchart of the graph rooted here.
- `op.get_dependencies(recursive=True, include_names=True)` — returns `[(field_path, dep), ...]` so you can see how each upstream node is referenced.
- `op.get_parameters()` — pulls just the scalar fields (anything not OpSpec-typed). Handy for grouping/aggregating results across a sweep.
- `op.has_result()` — cheap existence check against the store. Doesn't compute. Combine with `prepare()` to drive remote workflows.
- `UUIDMismatchError` almost always means: you added/removed a field on an OpSpec, or changed its default, and an old serialized graph now reconstructs to a different UUID. **This is almost always a good thing**, because it prevents changing implementations from breaking static assumptions. Fields that should *not* invalidate hashes when changed can be annotated `Annotated[T, ExcludeFromUUID()]`. To force hash invalidation, add a `_version: str = "todays_date"` field that you can manually bump when the source code implementation changes.
