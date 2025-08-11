# Mixin types for various runtime objects
from krnel.graph.op_spec import OpSpec


class DatasetType(OpSpec):
    content_hash: str

    def col_embedding(self, column_name: str) -> 'VectorColumnType':
        from krnel.graph.dataset_ops import SelectEmbeddingColumnOp
        return SelectEmbeddingColumnOp(column_name=column_name, dataset=self)

    def col_train_test_split(self, column_name: str) -> 'TrainTestSplitColumnType':
        from krnel.graph.dataset_ops import SelectTrainTestSplitColumnOp
        return SelectTrainTestSplitColumnOp(column_name=column_name, dataset=self)

    def col_prompt(self, column_name: str) -> 'TextColumnType':
        from krnel.graph.dataset_ops import SelectTextColumnOp
        return SelectTextColumnOp(column_name=column_name, dataset=self)

    def col_categorical(self, column_name: str) -> 'CategoricalColumnType':
        from krnel.graph.dataset_ops import SelectCategoricalColumnOp
        return SelectCategoricalColumnOp(column_name=column_name, dataset=self)

    def make_train_test_split(
        self,
        hash_column: 'TextColumnType',
        test_size: float | int | None = None,
        train_size: float | int | None = None,
        random_state: int = 42
    ) -> 'TrainTestSplitColumnType':
        from krnel.graph.dataset_ops import AssignTrainTestSplitOp
        return AssignTrainTestSplitOp(
                hash_column=hash_column,
                test_size=test_size,
                train_size=train_size,
                random_state=random_state
        )
    def template(self, template: str, **context: 'TextColumnType') -> 'TextColumnType':
        from krnel.graph.dataset_ops import JinjaTemplatizeOp
        return JinjaTemplatizeOp(template=template, context=context)

    def take(self, num_rows: int | None = None, *, skip: int = 1) -> 'DatasetType':
        from krnel.graph.dataset_ops import TakeRowsOp
        return TakeRowsOp(
            dataset=self,
            num_rows=num_rows,
            skip=skip,
            content_hash=self.content_hash + f".take({num_rows}, {skip})"
        )

class VectorColumnType(OpSpec):
    def train_classifier(
        self,
        model_name: str,
        labels: 'CategoricalColumnType',
        train_test_split: 'TrainTestSplitColumnType',
    ) -> 'ClassifierType':
        from krnel.graph.classifier_ops import TrainClassifierOp
        return TrainClassifierOp(
            model_name=model_name,
            x=self,
            y=labels,
            train_test_split=train_test_split,
        )
    def umap_vis(
        self,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        n_epochs: int = 200,
        random_state: int = 42,
    ) -> 'VizEmbeddingColumnType':
        from krnel.graph.viz_ops import UMAPVizOp
        return UMAPVizOp(
            input_embedding=self,
            n_epochs=n_epochs,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
        )

class VizEmbeddingColumnType(OpSpec):
    ...

class ClassifierType(OpSpec):
    def predict(self, input_data: 'VectorColumnType') -> 'ScoreColumnType':
        from krnel.graph.classifier_ops import ClassifierPredictOp
        return ClassifierPredictOp(
            model=self,
            x=input_data,
        )

class TextColumnType(OpSpec):
    def llm_generate_text(self, *, model_name: str, max_tokens: int = 100) -> 'TextColumnType':
        from krnel.graph.llm_ops import LLMGenerateTextOp
        return LLMGenerateTextOp(
            model_name=model_name,
            prompt=self,
            max_tokens=max_tokens,
        )

    def llm_layer_activations(
        self,
        *,
        model_name: str,
        layer_num: int,
        token_mode: str,
        batch_size: int,
        dtype: str | None = None,
        max_length: int | None = None,
        device: str = "auto",
    ) -> VectorColumnType:
        from krnel.graph.llm_ops import LLMLayerActivationsOp
        return LLMLayerActivationsOp(
            model_name=model_name,
            text=self,
            layer_num=layer_num,
            token_mode=token_mode,
            dtype=dtype,

            batch_size=batch_size,
            max_length=max_length,
            device=device,
        )


class ConversationColumnType(OpSpec): ...
class CategoricalColumnType(OpSpec): ...
class TrainTestSplitColumnType(OpSpec): ...
class ScoreColumnType(OpSpec): ...
