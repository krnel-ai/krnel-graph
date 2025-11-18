Functions and types reference
=============================
.. currentmodule:: krnel.graph.types

:class:`~krnel.graph.types.DatasetType` quick reference
-------------------------------------------------------

.. list-table::
    :header-rows: 1
    :width: 100%

    * - Method
      -
      - Return type
    * - :meth:`~DatasetType.assign_train_test_split()`
      - Assign a train / test split to the dataset.
      - :class:`TrainTestSplitColumnType`
    * - :meth:`~DatasetType.col_boolean()`
      - Get an existing column.
      - :class:`BooleanColumnType`
    * - :meth:`~DatasetType.col_categorical()`
      - ⋮
      - :class:`CategoricalColumnType`
    * - :meth:`~DatasetType.col_conversation()`
      - ⋮
      - :class:`ConversationColumnType`
    * - :meth:`~DatasetType.col_json()`
      - ⋮
      - :class:`JSONColumnType`
    * - :meth:`~DatasetType.col_score()`
      - ⋮
      - :class:`ScoreColumnType`
    * - :meth:`~DatasetType.col_text()`
      - ⋮
      - :class:`TextColumnType`
    * - :meth:`~DatasetType.col_train_test_split()`
      - ⋮
      - :class:`TrainTestSplitColumnType`
    * - :meth:`~DatasetType.col_vector()`
      - ⋮
      - :class:`VectorColumnType`
    * - :meth:`~DatasetType.mask_rows()`
      - Take a subset based on a boolean mask
      - :class:`DatasetType`
    * - :meth:`~DatasetType.take()`
      - Sample rows from the dataset.
      - :class:`DatasetType`


:class:`~krnel.graph.types.TextColumnType` quick reference
----------------------------------------------------------

.. list-table::
    :header-rows: 1
    :width: 100%

    * - Method
      - Description
      - Return type
    * - :meth:`~TextColumnType.llm_generate_text`
      - Generate text using a language model.
      - :class:`TextColumnType`
    * - :meth:`~TextColumnType.llm_layer_activations`
      - Fetch layer activations from a certain layer of  language model.
      - :class:`VectorColumnType`
    * - :meth:`~TextColumnType.llm_logit_scores`
      - Fetch logit scores for specified tokens from a language model.
      - :class:`VectorColumnType`
    * - :meth:`~TextColumnType.parse_json`
      - Parse JSON strings from this text column into structured JSON data.
      - :class:`JSONColumnType`
    * - :meth:`~CategoricalColumnType.is_in`
      - Check if text **is** present in a list of possibilities.
      - :class:`BooleanColumnType`
    * - :meth:`~CategoricalColumnType.not_in`
      - Check if text **is not** present in a list of possibilities.
      - :class:`BooleanColumnType`

:class:`~krnel.graph.types.VectorColumnType` quick reference
------------------------------------------------------------

.. list-table::
    :header-rows: 1
    :width: 100%

    * - Method
      - Description
      - Return type
    * - :meth:`~VectorColumnType.col`
      - Select a certain element of the vector.
      - :class:`ScoreColumnType`
    * - :meth:`~VectorColumnType.train_classifier`
      - Train a probe classifier on this vector.
      - :class:`ClassifierType`
    * - :meth:`~VectorColumnType.umap_vis`
      - Create a UMAP visualization from vectors.
      - :class:`VizEmbeddingColumnType`

:class:`~ClassifierType` quick reference
-----------------------------------------

.. list-table::
    :header-rows: 1
    :width: 100%

    * - Method
      - Description
      - Return type
    * - :meth:`~ClassifierType.predict`
      - Run this classifier on new data.
      - :class:`ScoreColumnType`

:class:`~CategoricalColumnType` quick reference
-------------------------------------------------

.. list-table::
    :header-rows: 1
    :width: 100%

    * - Method
      - Description
      - Return type
    * - :meth:`~CategoricalColumnType.is_in`
      - Check if category **is** in a list of categories.
      - :class:`BooleanColumnType`
    * - :meth:`~CategoricalColumnType.not_in`
      - Check if category **is not** in a list of categories.
      - :class:`BooleanColumnType`

:class:`~ScoreColumnType` quick reference
-------------------------------------------

.. list-table::
    :header-rows: 1
    :width: 100%

    * - Method
      - Description
      - Return type
    * - :meth:`~ScoreColumnType.evaluate`
      - Generate metrics for binary classification based on these scores.
      - :class:`EvaluationReportType`

Dataset types
-------------

.. autopydantic_model:: DatasetType

Column types
------------

.. autopydantic_model:: TextColumnType
.. autopydantic_model:: RowIDColumnType
.. autopydantic_model:: BooleanColumnType
.. autopydantic_model:: VizEmbeddingColumnType
.. autopydantic_model:: VectorColumnType
.. autopydantic_model:: ClassifierType
.. autopydantic_model:: CategoricalColumnType
.. autopydantic_model:: ConversationColumnType
.. autopydantic_model:: JSONColumnType
.. autopydantic_model:: EvaluationReportType
.. autopydantic_model:: ScoreColumnType
.. autopydantic_model:: TrainTestSplitColumnType