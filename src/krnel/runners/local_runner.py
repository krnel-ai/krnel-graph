# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Annotated
import warnings

from krnel.graph import SelectColumnOp
from krnel.graph.classifier_ops import TrainClassifierOp
from krnel.graph.dataset_ops import LoadDatasetOp, SelectCategoricalColumnOp, SelectVectorColumnOp, SelectTextColumnOp, SelectTrainTestSplitColumnOp, TakeRowsOp, FromListOp
from krnel.graph.llm_ops import LLMLayerActivationsOp
from krnel.graph.op_spec import OpSpec, graph_deserialize, graph_serialize, ExcludeFromUUID
from krnel.graph.grouped_ops import GroupedOp
from krnel.graph.types import DatasetType
from krnel.graph.viz_ops import UMAPVizOp
from krnel.runners.base_runner import BaseRunner, DontSave

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import fsspec

from krnel.runners.op_status import OpStatus
from krnel.runners.materialized_result import MaterializedResult
from krnel.runners.model_registry import get_layer_activations

_RESULT_PQ_FILE_SUFFIX = 'result.parquet'
_STATUS_JSON_FILE_SUFFIX = 'status.json'

class LoadLocalParquetDatasetOp(LoadDatasetOp):
    file_path: Annotated[str, ExcludeFromUUID()]
    # This op already includes a sha256sum of the actual
    # dataset, so we don't need to include the file_path.
    # In particular, this allows this dataset
    # to be materialized on a different machine
    # where the file_path may be different.

class LocalArrowRunner(BaseRunner):
    """
    A runner that executes operations locally and caches results as Arrow Parquet files.

    """
    def __init__(
        self,
        store_uri: str | None = None,
        filesystem: fsspec.AbstractFileSystem | str | None = None,
    ):
        """initialize runner with an fsspec filesystem and a base path within it.

        - if only root_path is provided (e.g., "s3://bucket/prefix" or "/tmp/krnel"), infer fs via fsspec.
        - if filesystem is provided, root_path should be a path valid for that fs (protocol will be stripped if present).
        - defaults to in-memory fs when nothing given.
        """
        if filesystem is None:
            if store_uri is None:
                store_uri = "memory://"
                warnings.warn("No store_uri specified. Will recompute results every time. "
                              "Use store_uri='memory://' to avoid this warning.")
            fs, _token, paths = fsspec.get_fs_token_paths(store_uri)
            base_path = paths[0]
        else:
            if isinstance(filesystem, str):
                fs = fsspec.filesystem(filesystem)
            else:
                fs = filesystem
            if store_uri is None:
                store_uri = ''
            if ':' in store_uri:
                raise ValueError("store_uri should not include a protocol prefix when filesystem is provided")
            base_path = store_uri
        # normalize trailing separators
        self.fs: fsspec.AbstractFileSystem = fs
        self.store_path_base: str = base_path.rstrip(fs.sep)

        # Which datasets have been materialized
        self._materialized_datasets = set()
        # Materializing datasets ourselves is important because remote
        # runners may not have access to the same files.

    def _join(self, *parts: str) -> str:
        """Join parts into a path, ensuring no double separators."""
        cleaned = []
        for i, p in enumerate(parts):
            if not p:
                continue
            s = str(p)
            if i > 0:
                s = s.lstrip(self.fs.sep)
            s = s.rstrip(self.fs.sep)
            cleaned.append(s)
        return self.fs.sep.join(cleaned) if cleaned else ""

    def _path(self, spec: OpSpec | str, extension: str) -> str:
        """Generate a path prefix for the given OpSpec and file extension."""
        if isinstance(spec, str):
            classname, hash = OpSpec.parse_uuid(spec)
            uuid = spec
        else:
            classname = spec.__class__.__name__
            uuid = spec.uuid
        dir_path = self._join(
            self.store_path_base,
            classname,
            # spec.uuid_hash[:2]
        )
        file_path = self._join(dir_path, f"{uuid}.{extension}")
        self.fs.makedirs(dir_path, exist_ok=True)
        return file_path

    def from_parquet(self, path: str) -> LoadLocalParquetDatasetOp:
        """Create a LoadParquetDatasetOp from a Parquet file path (local or remote)."""
        # compute content hash by streaming bytes; fsspec.open infers the fs from the URL
        h = sha256()
        with fsspec.open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return LoadLocalParquetDatasetOp(
            content_hash=h.hexdigest(),
            file_path=path,
        )

    def prepare(self, spec: OpSpec) -> None:
        """
        Materialize root dataset(s) up front to ensure they're in the backing store.

        This is particularly important for LoadLocalParquetDatasetOp, which may reference files
        that are not accessible on remote runners.
        """
        super().prepare(spec)
        for dataset in spec.get_dependencies(True):
            if isinstance(dataset, LoadLocalParquetDatasetOp):
                if dataset.uuid not in self._materialized_datasets:
                    if not self.has_result(dataset):
                        self.materialize(dataset)
                self._materialized_datasets.add(dataset.uuid)

    def from_list(self, data: dict[str, list[Any]]) -> FromListOp:
        """Create a FromListOp from Python lists/dicts."""
        return FromListOp(
            content_hash=sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(),
            data=data,
        )

    def get_result(self, spec: OpSpec) -> pa.Table:
        path = self._path(spec, _RESULT_PQ_FILE_SUFFIX)
        with self.fs.open(path, "rb") as f:
            table = pq.read_table(f)
        return MaterializedResult.from_any(table, spec)

    def has_result(self, spec: OpSpec) -> bool:
        return self.fs.exists(self._path(spec, _RESULT_PQ_FILE_SUFFIX))

    def put_result(self, spec: OpSpec, result: Any) -> bool:
        path = self._path(spec, _RESULT_PQ_FILE_SUFFIX)
        table = result.to_arrow()
        with self.fs.open(path, "wb") as f:
            pq.write_table(table, f)
        return True

    def uuid_to_op(self, uuid: str) -> OpSpec | None:
        path = self._path(uuid, _STATUS_JSON_FILE_SUFFIX)
        if self.fs.exists(path):
            with self.fs.open(path, "rt") as f:
                text = f.read()
            result = json.loads(text)
            results = graph_deserialize(result['op'])
            return results[0]
        return None

    def get_status(self, spec: OpSpec) -> OpStatus:
        path = self._path(spec, _STATUS_JSON_FILE_SUFFIX)
        if self.fs.exists(path):
            with self.fs.open(path, "rt") as f:
                text = f.read()
            result = json.loads(text)
            # Need to deserialize OpSpec separately
            [result['op']] = graph_deserialize(result['op'])
            status = OpStatus.model_validate(result)
            return status
        return OpStatus(op=spec, state='unsubmitted')

    def put_status(self, status: OpStatus) -> bool:
        path = self._path(status.op, _STATUS_JSON_FILE_SUFFIX)
        with self.fs.open(path, "wt") as f:
            f.write(status.model_dump_json())
        return True


@LocalArrowRunner.implementation
def load_parquet_dataset(runner, op: LoadLocalParquetDatasetOp):
    with fsspec.open(op.file_path, "rb") as f:
        return pq.read_table(f)


@LocalArrowRunner.implementation
def select_column(runner, op: SelectColumnOp | SelectTextColumnOp | SelectTrainTestSplitColumnOp | SelectVectorColumnOp | SelectCategoricalColumnOp):
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
def make_umap_viz(runner, op: UMAPVizOp):
    import umap
    dataset = runner.materialize(op.input_embedding).to_numpy().astype(np.float32)
    kwds = op.model_dump()
    del kwds['type']
    del kwds['input_embedding']
    reducer = umap.UMAP(verbose=True, **kwds)
    embedding = reducer.fit_transform(dataset)
    return embedding


@LocalArrowRunner.implementation
def registry_get_layer_activations(runner, op: LLMLayerActivationsOp):
    """LLM embedding using the model registry for dispatching."""
    # Use model registry to dispatch based on model_name URL
    return get_layer_activations(runner, op)


@LocalArrowRunner.implementation
def from_list_dataset(runner, op: FromListOp):
    """Convert Python list data to Arrow table."""
    return pa.table(op.data)


@LocalArrowRunner.implementation
def grouped_op(runner, op: GroupedOp):
    """Run a GroupedOp by running each op in sequence and returning the last result."""
    result = None
    for sub_op in op.ops:
        result = runner.materialize(sub_op)
    return result