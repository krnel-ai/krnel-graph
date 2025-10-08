Index of mechanistic interpretability operations
================================================

.. toctree::
   :maxdepth: 2


Classifiers and probes
----------------------

.. currentmodule:: krnel.graph.classifier_ops
.. autopydantic_model:: TrainClassifierOp
.. autopydantic_model:: ClassifierPredictOp
.. autopydantic_model:: ClassifierEvaluationOp

LLM and embedding operations
----------------------------

.. currentmodule:: krnel.graph.llm_ops
.. autopydantic_model:: LLMGenerateTextOp
.. autopydantic_model:: LLMLayerActivationsOp
.. autopydantic_model:: LLMLogitScoresOp


Basic dataset operations
------------------------
.. currentmodule:: krnel.graph.dataset_ops
.. autopydantic_model:: LoadDatasetOp
.. autopydantic_model:: LoadInlineJsonDatasetOp
.. autopydantic_model:: LoadLocalParquetDatasetOp
.. autopydantic_model:: SelectBooleanColumnOp
.. autopydantic_model:: SelectCategoricalColumnOp
.. autopydantic_model:: SelectColumnOp
.. autopydantic_model:: SelectConversationColumnOp
.. autopydantic_model:: SelectScoreColumnOp
.. autopydantic_model:: SelectTextColumnOp
.. autopydantic_model:: SelectTrainTestSplitColumnOp
.. autopydantic_model:: SelectVectorColumnOp
.. autopydantic_model:: TakeRowsOp
.. autopydantic_model:: MaskRowsOp
.. .. autopydantic_model:: AssignRowIDOp
.. autopydantic_model:: AssignTrainTestSplitOp

Boolean operations
------------------
.. autopydantic_model:: BooleanLogicOp
.. autopydantic_model:: CategoryToBooleanOp

Text operations
---------------
.. autopydantic_model:: JinjaTemplatizeOp

Score column operations
-----------------------
.. autopydantic_model:: PairwiseArithmeticOp
.. autopydantic_model:: VectorToScalarOp


Control flow operations
-----------------------
.. currentmodule:: krnel.graph.group_ops
.. autopydantic_model:: GroupedOp

Visualization operations
------------------------
.. currentmodule:: krnel.graph.viz_ops
.. autopydantic_model:: UMAPVizOp