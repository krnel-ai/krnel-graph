krnel Documentation
===================

Welcome to **krnel**, a powerful client library for building and executing computation graphs with machine learning operations, dataset transformations, and visualization tools.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting-started
   examples
   api/index

Features
--------

* **Type-safe computation graphs**: Build complex ML pipelines with strong typing
* **Dataset operations**: Load, transform, and manipulate datasets with ease
* **Machine learning**: Train classifiers, extract LLM activations, generate text
* **Visualization**: Create UMAP embeddings and interactive visualizations
* **Flexible execution**: Run locally or on distributed systems

Quick Start
-----------

Install krnel with optional dependencies:

.. code-block:: bash

   pip install krnel[viz,cli]

Create your first computation graph:

.. code-block:: python

   import krnel

   # Load a dataset
   dataset = krnel.load_dataset("my_data.parquet")

   # Extract text and create embeddings
   text_col = dataset.col_prompt("text_column")
   embeddings = text_col.llm_layer_activations(
       model_name="bert-base-uncased",
       layer_num=-1,
       token_mode="mean",
       batch_size=32
   )

   # Create visualization
   viz = embeddings.umap_vis(n_neighbors=15, min_dist=0.1)

   # Execute the computation graph
   result = krnel.run(viz)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`