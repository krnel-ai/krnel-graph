from hashlib import sha256
from pathlib import Path
from typing import Any

from krnel.graph import SelectColumnOp
from krnel.graph.classifier_ops import TrainClassifierOp
from krnel.graph.dataset_ops import LoadDatasetOp, SelectCategoricalColumnOp, SelectEmbeddingColumnOp, SelectPromptColumnOp, SelectTrainTestSplitColumnOp
from krnel.graph.llm_ops import LLMEmbedOp
from krnel.graph.op_spec import OpSpec
from krnel.graph.types import DatasetType
from krnel.runners.base_runner import BaseRunner, DontSave

import pyarrow as pa
import pyarrow.parquet as pq

from krnel.runners.op_status import OpStatus

class LoadLocalParquetDatasetOp(LoadDatasetOp):
    file_path: str

class LocalArrowRunner(BaseRunner):
    def __init__(self, cache_folder: str):
        self.cache_folder = cache_folder

    def _path(self, spec: OpSpec, extension: str) -> Path:
        path = (Path(self.cache_folder)
                / spec.uuid[-2:]
                / f"{spec.uuid}.{extension}")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def from_parquet(self, path: str) -> LoadLocalParquetDatasetOp:
        """Create a LoadParquetDatasetOp from a Parquet file path."""
        return LoadLocalParquetDatasetOp(
            content_hash=sha256(Path(path).read_bytes()).hexdigest(),
            file_path=path,
        )

    def get_result(self, spec: OpSpec) -> Any:
        path = self._path(spec, 'result.parquet')
        return pq.read_table(path)

    def _validate_result(self, spec: OpSpec, result: Any) -> Any | bool:
        if isinstance(result, pa.Table):
            return result
        elif isinstance(result, (list, dict)):
            return pa.Table.from_pydict(result)
        elif isinstance(result, pa.Array):
            return pa.Table.from_arrays([result], names=[spec.uuid])
        else:
            raise ValueError(f"Result of {spec} is not a valid Arrow Table: {result}")

    def put_result(self, spec: OpSpec, result: Any) -> bool:
        path = self._path(spec, 'result.parquet')
        pq.write_table(result, path)
        return True

    def get_status(self, spec: OpSpec) -> OpStatus:
        path = self._path(spec, 'status.json')
        return OpStatus.model_validate_json(path.read_text()) if path.exists() else OpStatus(
            op=spec,
            state='pending',
        )
    def put_status(self, status: OpStatus) -> bool:
        path = self._path(status.op, 'status.json')
        path.write_text(status.model_dump_json())
        return True


@LocalArrowRunner.implementation
def load_parquet_dataset(runner, op: LoadLocalParquetDatasetOp):
    return pq.read_table(op.file_path)


@LocalArrowRunner.implementation
def select_column(runner, op: SelectColumnOp | SelectPromptColumnOp | SelectTrainTestSplitColumnOp | SelectEmbeddingColumnOp | SelectCategoricalColumnOp):
    dataset = runner.materialize(op.dataset)
    return DontSave(dataset[op.column_name])

