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