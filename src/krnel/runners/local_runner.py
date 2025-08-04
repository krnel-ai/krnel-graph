from krnel.graph import SelectColumnOp
from krnel.graph.dataset_ops import LoadDatasetOp
from krnel.graph.types import DatasetType
from krnel.runners.base_runner import BaseRunner

class LoadLocalParquetDatasetOp(LoadDatasetOp):
    file_path: str

class LocalFolderRunner(BaseRunner):
    def __init__(self, cache_folder: str):
        self.cache_folder = cache_folder

    def from_parquet(self, path: str) -> LoadLocalParquetDatasetOp:
        """Create a LoadParquetDatasetOp from a Parquet file path."""
        return LoadLocalParquetDatasetOp(content_hash="HASH HERE", file_path=path)

@LocalFolderRunner.implementation(LoadLocalParquetDatasetOp)
def load_parquet_dataset(runner, spec: LoadLocalParquetDatasetOp):
    print("Loading Parquet dataset from:", spec.file_path)
    print("Cache folder:", runner.cache_folder)
    print(spec)
    return "LOADED DATASET"

@LocalFolderRunner.implementation(SelectColumnOp)
def select_column(runner, spec: SelectColumnOp):
    print("Selecting column:", spec.column_name)
    print("From dataset with content hash:", spec.dataset.content_hash)
    print("Materializing parent:", runner.materialize(spec.dataset))
    return spec


runner = LocalFolderRunner(cache_folder='/tmp/cache')

dataset = runner.from_parquet('/tmp/data.parquet')

embeddings = dataset.col_prompt('prompt_column').llm_embed(
    model_name='hf:gpt-3.5-turbo',
    layer_num=-1,
    token_mode='last',
)

model = embeddings.train_classifier(
    model_name="my_model",
    labels=dataset.col_categorical('label_column'),
    train_test_split=dataset.col_train_test_split('train_test_split'),
)


print(repr(model))
print(model.model_dump_for_uuid())
print(model.get_parents(recursive=True))
print(model.get_parents(recursive=True, of_type=LoadDatasetOp))
