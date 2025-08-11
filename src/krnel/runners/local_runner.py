from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from krnel.graph import SelectColumnOp
from krnel.graph.classifier_ops import TrainClassifierOp
from krnel.graph.dataset_ops import LoadDatasetOp, SelectCategoricalColumnOp, SelectEmbeddingColumnOp, SelectTextColumnOp, SelectTrainTestSplitColumnOp, TakeRowsOp, FromListOp
from krnel.graph.llm_ops import LLMLayerActivationsOp
from krnel.graph.op_spec import OpSpec, graph_deserialize, graph_serialize
from krnel.graph.types import DatasetType
from krnel.graph.viz_ops import UMAPVizOp
from krnel.runners.base_runner import BaseRunner, DontSave

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from krnel.runners.op_status import OpStatus
from krnel.runners.materialized_result import MaterializedResult
from krnel.runners.model_registry import embed

_RESULT_PQ_FILE_SUFFIX = 'result.parquet'
_STATUS_JSON_FILE_SUFFIX = 'status.json'

class LoadLocalParquetDatasetOp(LoadDatasetOp):
    file_path: str

class LocalArrowRunner(BaseRunner):
    """
    A runner that executes operations locally and caches results as Arrow Parquet files.

    """
    def __init__(self, cache_folder: str):
        self.cache_folder = cache_folder

    def _path(self, spec: OpSpec, extension: str) -> Path:
        path = (Path(self.cache_folder)
                / spec.__class__.__name__
                / spec.uuid_hash[:2] # first two letters of the actual hash
                / f"{spec.uuid}.{extension}")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def from_parquet(self, path: str) -> LoadLocalParquetDatasetOp:
        """Create a LoadParquetDatasetOp from a Parquet file path."""
        return LoadLocalParquetDatasetOp(
            content_hash=sha256(Path(path).read_bytes()).hexdigest(),
            file_path=path,
        )

    def from_list(self, data: dict[str, list[Any]]) -> FromListOp:
        """Create a FromListOp from Python lists/dicts."""
        return FromListOp(
            content_hash=sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(),
            data=data,
        )

    def get_result(self, spec: OpSpec) -> pa.Table:
        """Return the result of the given OpSpec as a pyarrow Table."""
        path = self._path(spec, _RESULT_PQ_FILE_SUFFIX)
        return MaterializedResult.from_any(pq.read_table(path), spec)

    def has_result(self, spec: OpSpec) -> bool:
        """Returns True if the result for the given OpSpec exists."""
        return self._path(spec, _RESULT_PQ_FILE_SUFFIX).exists()

    def put_result(self, spec: OpSpec, result: Any) -> bool:
        path = self._path(spec, _RESULT_PQ_FILE_SUFFIX)
        result = result.to_arrow()
        pq.write_table(result, path)
        return True

    def get_status(self, spec: OpSpec) -> OpStatus:
        path = self._path(spec, _STATUS_JSON_FILE_SUFFIX)
        if path.exists():
            text = path.read_text()
            result = json.loads(text)
            # Need to deserialize OpSpec separately
            [result['op']] = graph_deserialize(result['op'])
            status = OpStatus.model_validate(result)
            return status
        return OpStatus(op=spec, state='unsubmitted')

    def put_status(self, status: OpStatus) -> bool:
        path = self._path(status.op, _STATUS_JSON_FILE_SUFFIX)
        path.write_text(status.model_dump_json())
        return True


@LocalArrowRunner.implementation
def load_parquet_dataset(runner, op: LoadLocalParquetDatasetOp):
    return pq.read_table(op.file_path)


@LocalArrowRunner.implementation
def select_column(runner, op: SelectColumnOp | SelectTextColumnOp | SelectTrainTestSplitColumnOp | SelectEmbeddingColumnOp | SelectCategoricalColumnOp):
    dataset = runner.materialize(op.dataset).to_arrow()
    return DontSave(dataset[op.column_name])

@LocalArrowRunner.implementation
def take_rows(runner, op: TakeRowsOp):
    table = runner.materialize(op.dataset).to_arrow()
    table = table[::op.skip]
    if op.num_rows is not None:
        return DontSave(table[:op.num_rows])
    return DontSave(table)


@LocalArrowRunner.implementation
def make_umap_embedding(runner, op: UMAPVizOp):
    import umap
    dataset = runner.materialize(op.input_embedding).to_numpy().astype(np.float32)
    kwds = op.model_dump()
    del kwds['type']
    del kwds['input_embedding']
    reducer = umap.UMAP(verbose=True, **kwds)
    embedding = reducer.fit_transform(dataset)
    return embedding


@LocalArrowRunner.implementation
def registry_llm_embed(runner, op: LLMLayerActivationsOp):
    """LLM embedding using the model registry for dispatching."""
    # Use model registry to dispatch based on model_name URL
    return embed(runner, op)


@LocalArrowRunner.implementation
def from_list_dataset(runner, op: FromListOp):
    """Convert Python list data to Arrow table."""
    return pa.table(op.data)