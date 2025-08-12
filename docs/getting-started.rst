Getting Started
===============

Installation
------------

Basic installation:

.. code-block:: bash

   pip install krnel

With optional dependencies for visualization and CLI:

.. code-block:: bash

   pip install krnel[viz,cli]

For development with testing:

.. code-block:: bash

   pip install krnel[test,viz,cli]

Core Concepts
-------------

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

Next Steps
----------

* See :doc:`examples` for complete working examples
* Browse the :doc:`api/index` for detailed API documentation
* Check out the CLI tools with ``krnel --help``