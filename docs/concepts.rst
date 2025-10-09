Core concepts
=============

Getting Started
---------------

Install krnel with optional dependencies:

.. code-block:: bash

   uv add krnel[ml,cli]

Create your first computation graph:

.. code-block:: python

   # TODO(kwilber): Add a simple example here

Philosophy
----------

krnel is built around **computation graphs** where each node represents an operation on data. The library provides type-safe operations for:

* **Datasets**: Loading and transforming tabular data
* **Columns**: Typed operations on specific data types (text, vectors, categories)
* **Machine Learning**: Training classifiers, extracting embeddings, generating text
* **Visualization**: Creating 2D projections of high-dimensional data

Basic Workflow
--------------

1. **Load Data**: Start with a dataset (Parquet, CSV, etc.)
2. **Select Columns**: Extract specific columns with appropriate types
3. **Transform**: Apply operations like embeddings, splits, templates
4. **Model/Visualize**: Train models or create visualizations
5. **Execute**: Run the computation graph to get results

Type System
-----------

krnel uses a rich type system to ensure operations are valid:

* ``DatasetType``: Represents tabular data with multiple columns
* ``TextColumnType``: Text data suitable for NLP operations
* ``VectorColumnType``: Numerical arrays/embeddings for ML
* ``CategoricalColumnType``: Discrete categories for classification
* ``TrainTestSplitColumnType``: Boolean indicators for ML splits
* ``ClassifierType``: Trained machine learning models

TODO(kwilber)

Think of Krnel-graph as **"git for ML data transformations"** - every operation has a content hash of its parameters and dependencies. Results are cached and you can reproduce any computation exactly. Because the graph is strongly typed, **all operations are serializable and easily discoverable**, by you, your editor, and the agents you use.

Krnel-graph separates specification from implementation. Each operation's definition contains everything needed to materialize that operation, and each Runner can implement each operation differently. This lets you **swap in different backends**, dataflow executors, orchestrators, etc.

What this library is
--------------------

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

What this library is not
------------------------

- ❌ ...a **task orchestrator** like Airflow or Prefect
    - No YAML templates, no Docker containers (by default)
- ❌ ...a **distributed computing framework** like Dask or Ray
    - The default runner uses local-only execution for now
    - Results can be saved and loaded to a remote store (NFS, GCS/S3, ...)
    - Bring your own scheduling / workflow management if needed
- ❌ ...an **experimentation or visualization tool** (though it integrates nicely with notebooks and plotting libraries)

The goal of krnel-graph is to separate well-typed specifications from their implementation. Krnel-graph does not depend on particular infrastructure. All operations are separated from their implementations, so it's easy to swap in your own dataflow executor if you prefer.

Core Concepts
------------------------

:obj:`~krnel.graph.op_spec.OpSpec`: Content-Addressable Operations
******************************************************************

Every operation in Krnel is an `OpSpec` - an immutable specification with a deterministic UUID::

   from krnel.graph import LoadInlineJsonDatasetOp

   # These two operations have identical UUIDs
   op1 = LoadInlineJsonDatasetOp(data={'x': [1, 2, 3]})
   op2 = LoadInlineJsonDatasetOp(data={'x': [1, 2, 3]})
   assert op1.uuid == op2.uuid

Krnel uses a type-driven fluent API where each column type provides relevant methods::

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

Runners: Execution Engines
****************************************

Runners execute your computation graphs.

.. code-block::
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

Writing custom operations
------------------------

1. Define your operation class

   .. code-block::

      from krnel.graph import OpSpec
      from krnel.graph.types import TextColumnType, VectorColumnType

      class MyCustomEmbeddingOp(VectorColumnType):
         """Extract embeddings using a custom model."""
         text_input: TextColumnType
         model_path: str
         max_length: int = 512

2. Implement the execution logic:

   .. code-block::

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

3. Use your custom operation:

   .. code-block::

      dataset = LoadInlineJsonDatasetOp(data={'text': ['Hello world', 'Custom ops!']})
      text_col = dataset.col_text('text')

      # Using your custom operation
      embeddings = text_col.my_custom_embedding(
         model_path='./my-model',
         max_length=256
      )

      result = runner.to_numpy(embeddings)