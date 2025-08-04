from pydantic import BaseModel

# Mixin types for various runtime objects
class DatasetType(BaseModel):
    content_hash: str

    def col_embedding(self, column_name: str) -> 'VectorColumnType':
        from krnel.graph.dataset_ops import SelectEmbeddingColumnOp
        return SelectEmbeddingColumnOp(column_name=column_name, dataset=self)

    def col_train_test_split(self, column_name: str) -> 'TrainTestSplitColumnType':
        from krnel.graph.dataset_ops import SelectTrainTestSplitColumnOp
        return SelectTrainTestSplitColumnOp(column_name=column_name, dataset=self)

    def col_prompt(self, column_name: str) -> 'TextColumnType':
        from krnel.graph.dataset_ops import SelectPromptColumnOp
        print("SELECT PROMPT COLUMN")
        return SelectPromptColumnOp(column_name=column_name, dataset=self)

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

class VectorColumnType(BaseModel):
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

class VizEmbeddingColumnType(BaseModel):
    ...

class ClassifierType(BaseModel):
    def predict(self, input_data: 'VectorColumnType') -> 'ScoreColumnType':
        from krnel.graph.classifier_ops import ClassifierPredictOp
        return ClassifierPredictOp(
            model=self,
            x=input_data,
        )

class TextColumnType(BaseModel):
    def llm_generate_text(self, model_name: str, max_tokens: int = 100) -> 'TextColumnType':
        from krnel.graph.llm_ops import LLMGenerateTextOp
        return LLMGenerateTextOp(
            model_name=model_name,
            prompt=self,
            max_tokens=max_tokens,
        )

    def llm_embed(self, model_name: str, layer_num: int, token_mode: str) -> VectorColumnType:
        from krnel.graph.llm_ops import LLMEmbedOp
        return LLMEmbedOp(
            model_name=model_name,
            text=self,
            layer_num=layer_num,
            token_mode=token_mode,
        )

class ConversationColumnType(BaseModel): ...
class CategoricalColumnType(BaseModel): ...
class TrainTestSplitColumnType(BaseModel): ...
class ScoreColumnType(BaseModel): ...
