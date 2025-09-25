# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

Krnel is a library for building content-addressable computation graphs. The architecture has three main layers:

### 1. Graph Layer (`src/krnel/graph/`)
- **OpSpec** (`op_spec.py`): Core abstraction representing immutable nodes in a computation DAG. Each OpSpec has a deterministic UUID based on its content and implements content-addressable semantics.
  - `dataset_ops.py`: Dataset loading, column selection, row operations
  - `classifier_ops.py`: Classification training operations
  - `llm_ops.py`: LLM-based operations and layer activations
  - `viz_ops.py`: Visualization operations (UMAP, etc.)
- **Operation Types**: Specialized OpSpec subclasses for different operations:
  - `types.py`: Entire user-facing fluent API lives here. (Client code should prefer these functions instead of instantiating `OpSpec`s manually)
- **Graph Transformations** (`graph_transformations.py`): Functions for DAG traversal, dependency analysis, and graph manipulation

### 2. Runners Layer (`src/krnel/graph/runners/`)
- **LocalArrowRunner** (`local_runner.py`): Executes operations locally, caching results as Arrow Parquet files using fsspec for storage abstraction
- **ModelProvider** (`model_registry.py`): Registry system for ML model providers and activation extraction
- **Result Conversion Methods**: Each runner provides `to_numpy()`, `to_arrow()`, `to_json()` methods for accessing computation results in different formats
- **OpStatus** (`op_status.py`): Tracks execution status of operations

### 3. CLI and Utilities
- **CLI** (`cli.py`): Command-line interface built with cyclopts for operation status checking
- **Visualization** (`viz/`): UMAP-based visualization utilities

## Key Concepts
- **Content-Addressable Operations**: Every OpSpec has a deterministic UUID computed from its content. Identical operations producing the same data always produce the same UUID.
- **DAG Semantics**: OpSpec fields referencing other OpSpecs create DAG edges. Fields are inputs / dependencies. Scalar fields are treated as parameters.
- **Immutability**: OpSpecs cannot be modified after creation. Changes create new OpSpecs.
- **ExcludeFromUUID**: Use `Annotated[Type, ExcludeFromUUID()]` to exclude fields from UUID computation while keeping them in serialization.
  - Use for parameters/details that are useful for provenance tracking but shouldn't affect results.
  - Ideal for local paths, ephemeral configuration that shouldn't affect results, etc.
  - Because ML frameworks are broadly nondeterministic, it's probably best to include batch size, device, and similar parameters inside UUID computation

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

Tests are in `tests/` directory using pytest. The project generally follows **Test-Driven Development (TDD)** philosophy - failing tests define expected behavior before implementation.

The test suite covers:
- OpSpec serialization/deserialization and UUID computation
- Graph transformation operations
- Runner execution and caching behavior
- LocalArrowRunner operations (`test_local_arrow_runner.py`)

### Testing Philosophy
- **Failing tests are specifications** - They define what needs to be implemented next
- **No skipping or special-casing unimplemented features** - Tests fail explicitly to show implementation gaps

Key test files:
- `test_graph.py` - Core OpSpec functionality
- `test_graph_transformations.py` - DAG manipulation
- `test_local_arrow_runner.py` - LocalArrowRunner operations (LoadInlineJsonDatasetOp, SelectColumnOp, TakeRowsOp, etc.)

Run individual test files or functions using pytest with the patterns shown in the commands section above.



## Optional Dependencies

The project has several optional dependency groups defined in `pyproject.toml`:
- `test`: Testing dependencies (pytest, pytest-cov)
- `viz`: Visualization dependencies (umap-learn, jupyter-scatter, seaborn, numba)
- `cli`: CLI dependencies (rich, cyclopts)
- `docs`: Documentation dependencies (sphinx, sphinx-autoapi, etc.)

Install with: `uv sync --extra <group_name>` or `--all-extras` for everything.