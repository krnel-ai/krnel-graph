Core concepts
=============

Getting Started
---------------

Install krnel with optional dependencies:

.. code-block:: bash

   uv add krnel[ml,cli]

Create your first computation graph:

.. code-block:: python

   import krnel

   # Load a dataset
   dataset = krnel.load_dataset("my_data.parquet")

   # Extract text and create embeddings
   text_col = dataset.col_text("text_column")
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

Example Types in Action
-----------------------

.. code-block:: python

   # Load dataset
   dataset = krnel.load_dataset("data.parquet")  # Returns DatasetType

   # Extract typed columns
   text = dataset.col_prompt("description")      # Returns TextColumnType
   labels = dataset.col_categorical("category")   # Returns CategoricalColumnType

   # Create embeddings
   embeddings = text.llm_layer_activations(      # Returns VectorColumnType
       model_name="sentence-transformers/all-MiniLM-L6-v2",
       layer_num=-1,
       token_mode="mean"
   )

   # Create train/test split
   split = dataset.make_train_test_split(         # Returns TrainTestSplitColumnType
       hash_column=text,
       test_size=0.2
   )

   # Train classifier
   classifier = embeddings.train_classifier(     # Returns ClassifierType
       model_name="logistic_regression",
       labels=labels,
       train_test_split=split
   )