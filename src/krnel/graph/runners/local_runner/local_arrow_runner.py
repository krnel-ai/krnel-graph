# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

import contextlib
import io
import json
import math
import os
import pickle
import random
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from krnel.graph import config
from krnel.graph.classifier_ops import ClassifierEvaluationOp
from krnel.graph.dataset_ops import (
    AssignTrainTestSplitOp,
    BooleanLogicOp,
    CategoryToBooleanOp,
    JinjaTemplatizeOp,
    LoadInlineJsonDatasetOp,
    LoadLocalParquetDatasetOp,
    MaskRowsOp,
    PairwiseArithmeticOp,
    SelectColumnOp,
    TakeRowsOp,
    VectorToScalarOp,
)
from krnel.graph.grouped_ops import GroupedOp
from krnel.graph.llm_ops import LLMLayerActivationsOp, LLMLogitScoresOp
from krnel.graph.op_spec import OpSpec, graph_deserialize
from krnel.graph.runners.base_runner import BaseRunner
from krnel.graph.runners.model_registry import get_layer_activations, get_llm_output_logits
from krnel.graph.runners.op_status import OpStatus
from krnel.graph.viz_ops import UMAPVizOp
from krnel.logging import get_logger

logger = get_logger(__name__)

# Global dictionary for result file formats
RESULT_FORMATS = {
    "arrow": "result.parquet",
    "json": "result.json",
    "pickle": "result.pickle",
}
RESULT_INDICATOR = "done"
STATUS_JSON_FILE_SUFFIX = "status.json"


class LocalArrowRunner(BaseRunner):
    """
    A runner that executes operations locally and caches results as Arrow Parquet files.

    """

    def __init__(
        self,
        store_uri: str | None = None,
        filesystem: fsspec.AbstractFileSystem | str | None = None,
    ):
        """Initialize runner with an fsspec filesystem and a base path within it.

        - if only root_path is provided (e.g., "s3://bucket/prefix" or "/tmp/krnel"), infer fs via fsspec.
        - if filesystem is provided, root_path should be a path valid for that fs (protocol will be stripped if present).
        - defaults to in-memory fs when nothing given.
        """
        self._materialization_cache = {}
        self._store_uri = store_uri
        if filesystem is None:
            if store_uri is None:
                store_uri = config.KrnelGraphConfig().store_uri
            fs, _token, paths = fsspec.get_fs_token_paths(store_uri)
            base_path = paths[0]
        else:
            if isinstance(filesystem, str):
                fs = fsspec.filesystem(filesystem)
            else:
                fs = filesystem
            if store_uri is None:
                raise ValueError("Must provide store_uri if filesystem is provided")
            if ":" in store_uri:
                raise ValueError(
                    "store_uri should not include a protocol prefix when filesystem is provided"
                )
            base_path = store_uri
        # normalize trailing separators
        self.fs: fsspec.AbstractFileSystem = fs
        self.store_path_base: str = base_path.rstrip(fs.sep)

        # Which datasets have been materialized
        self._materialized_datasets = set()
        # Materializing datasets ourselves is important because remote
        # runners may not have access to the same files.

    def _path(
        self,
        spec: OpSpec | str,
        basename: str,
        *,
        store_path_base: str | None = None,
        makedirs: bool = True,
    ) -> str:
        """Generate a path prefix for the given OpSpec and file extension."""
        if "/" in basename:
            raise ValueError(f"basename must not contain '/', {basename=}")
        if isinstance(spec, str):
            classname, uuid_hash_only = OpSpec.parse_uuid(spec)
        else:
            classname = spec.__class__.__name__
            uuid_hash_only = spec.uuid_hash
        dir_path = (
            Path(store_path_base or self.store_path_base) / classname / uuid_hash_only
        )
        if makedirs:
            self.fs.makedirs(str(dir_path), exist_ok=True)
        return str(dir_path / basename)

    @contextlib.contextmanager
    def _open_for_data(self, op: OpSpec, basename: str, mode: str) -> io.IOBase:
        "Context manager for opening data files."
        path = self._path(op, basename)
        log = logger.bind(path=path, mode=mode)
        log.debug("opening for data")
        with self.fs.open(path, mode) as f:
            yield f

    @contextlib.contextmanager
    def _open_for_status(self, op: OpSpec, basename: str, mode: str) -> io.IOBase:
        "Context manager for opening status files."
        path = self._path(op, basename)
        log = logger.bind(path=path, mode=mode)
        log.debug("opening for status")
        with self.fs.open(path, mode) as f:
            yield f

    def _finalize_result(self, op: OpSpec):
        "Mark a result as completed."
        done_path = self._path(op, RESULT_INDICATOR)
        log = logger.bind(path=done_path)
        log.debug("_finalize_result()")
        with self.fs.open(done_path, "wt") as f:
            f.write("done")

    def from_parquet(self, path: str, *, sha256sum: str | None = None) -> LoadLocalParquetDatasetOp:
        """An operation that loads a local Parquet dataset from a given path.

        Either the path must exist, or sha256sum must be provided to uniquely identify
        the file contents. If both are provided, the contents will be verified against
        the checksum.

        Providing ``sha256sum`` allows the operation to be used even if
        the file is not present, which is useful for remote runners.

        Arguments:
            path: The file path to the Parquet dataset.
            sha256sum: Optional expected sha256 checksum of the file contents.
                If provided, and if the path exists, the computed checksum will be verified against this value.

        Returns:
            A `LoadLocalParquetDatasetOp` representing the dataset.
        """
        log = logger.bind(path=path)
        if not os.path.exists(path) and sha256sum is None:
            raise ValueError(
                f"Path {path!r} does not exist and no sha256sum provided to verify contents."
            )
        if os.path.exists(path):
            # compute content hash by streaming bytes; fsspec.open infers the fs from the URL
            h = sha256()
            log.debug("Reading parquet dataset")
            # note: not using self.fs, because this is a local read
            with fsspec.open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            log.debug("Content hash", content_hash=h.hexdigest())
            if sha256sum is None:
                sha256sum = h.hexdigest()
            else:
                if h.hexdigest() != sha256sum:
                    raise ValueError(
                        f"Checksum mismatch for {path!r}: expected {sha256sum}, got {h.hexdigest()}"
                    )
        op = LoadLocalParquetDatasetOp(
            content_hash=sha256sum,
            file_path=path,
        )
        op._runner = self
        return op

    def prepare(self, op: OpSpec) -> None:
        """
        Materialize root dataset(s) up front to ensure they're in the backing store.

        This is particularly important for LoadLocalParquetDatasetOp, which may reference files
        that are not accessible on remote runners.
        """
        log = logger.bind(op=op.uuid)
        super().prepare(op)
        for dataset in op.get_dependencies(True):
            if isinstance(dataset, LoadLocalParquetDatasetOp):
                if dataset.uuid not in self._materialized_datasets:
                    if not self.has_result(dataset):
                        log.debug(
                            "prepare(): dataset needs materializing", dataset=dataset
                        )
                        self._materialize_if_needed(dataset)
                self._materialized_datasets.add(dataset.uuid)

    def from_inline_dataset(
        self, data: dict[str, list[Any]]
    ) -> LoadInlineJsonDatasetOp:
        """Create a LoadInlineJsonDatasetOp from Python lists/dicts."""
        op = LoadInlineJsonDatasetOp(
            content_hash=sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(),
            data=data,
        )
        op._runner = self
        return op

    def has_result(self, op: OpSpec) -> bool:
        if op.is_ephemeral:
            return True  # Ephemeral ops are always "available"

        # Check if any result format exists
        log = logger.bind(op=op.uuid)
        done_indicator = self._path(op, RESULT_INDICATOR)
        if self.fs.exists(done_indicator):
            log.debug("has_result()", result=True, path=done_indicator)
            return True
        # for format_name, suffix in _RESULT_FORMATS.items():
        #    path = self._path(spec, suffix)
        #    if self.fs.exists(path):
        #        log.debug("has_result()", result=True, format=format_name, path=path)
        #        return True

        log.debug("has_result()", result=False)
        return False

    def uuid_to_op(self, uuid: str) -> OpSpec | None:
        "Lookup a UUID by its name"
        log = logger.bind(uuid=uuid)
        path = self._path(uuid, STATUS_JSON_FILE_SUFFIX)
        if self.fs.exists(path):
            log.debug("uuid_to_op()", exists=True)
            with self._open_for_status(uuid, STATUS_JSON_FILE_SUFFIX, "rt") as f:
                text = f.read()
            result = json.loads(text)
            results = graph_deserialize(result["op"])
            results[0]._runner = self
            return results[0]
        log.debug("uuid_to_op()", exists=False)
        return None

    def get_status(self, op: OpSpec) -> OpStatus:
        if op.is_ephemeral:
            # Ephemeral ops do not have a status file, they are always 'ephemeral'
            return OpStatus(op=op, state="ephemeral")
        path = self._path(op.uuid, STATUS_JSON_FILE_SUFFIX)
        log = logger.bind(op=op.uuid)
        # log.debug("get_status()", stack_info=True)
        log.debug("get_status()")
        if self.fs.exists(path):
            with self._open_for_status(op, STATUS_JSON_FILE_SUFFIX, "rt") as f:
                result = json.load(f)
            # Need to deserialize OpSpec separately
            [result["op"]] = graph_deserialize(result["op"])
            status = OpStatus.model_validate(result)
            return status  # noqa: RET504
        else:
            log.debug("status not found, creating new")
            new_status = OpStatus(
                op=op, state="new" if not self.has_result(op) else "completed"
            )
            self.put_status(new_status)
            return new_status

    def put_status(self, status: OpStatus) -> bool:
        if status.op.is_ephemeral:
            # Ephemeral ops do not have a status file, they are always 'ephemeral'
            return True
        log = logger.bind(op=status.op.uuid)
        log.debug("put_status()", state=status.state)
        with self._open_for_status(status.op, STATUS_JSON_FILE_SUFFIX, "wt") as f:
            f.write(status.model_dump_json())
        return True

    # Implementation of BaseRunner abstract methods
    def to_arrow(self, op: OpSpec) -> pa.Table:
        log = logger.bind(op=op.uuid)
        if op.uuid in self._materialization_cache:
            cached_result = self._materialization_cache[op.uuid]
            if isinstance(cached_result, pa.Table):
                return cached_result
            else:
                raise ValueError(
                    "Result type doesn't match expected type for to_arrow()"
                )

        if self._materialize_if_needed(op):
            return self.to_arrow(op)  # load from cache

        log.debug("Loading arrow result from store")
        with self._open_for_data(op, RESULT_FORMATS["arrow"], "rb") as f:
            table = pq.read_table(f)
        self._materialization_cache[op.uuid] = table
        return table

    def to_pandas(self, op: OpSpec):
        table = self.to_arrow(op)
        return table.to_pandas()

    def to_numpy(self, op: OpSpec) -> np.ndarray:
        table = self.to_arrow(op)

        if table.num_columns == 1:
            return self._column_to_numpy(table.column(0))
        else:
            raise ValueError(
                f"to_numpy() expects single-column tables, got {table.num_columns} columns from {type(op).__name__}"
            )

    def to_json(self, op: OpSpec) -> dict:
        if op.uuid in self._materialization_cache:
            cached_result = self._materialization_cache[op.uuid]
            if isinstance(cached_result, dict):
                return cached_result
            else:
                raise ValueError(
                    "Result type doesn't match expected type for to_json()"
                )

        if self._materialize_if_needed(op):
            return self.to_json(op)  # load from cache

        with self._open_for_data(op, RESULT_FORMATS["json"], "rb") as f:
            import json

            result = json.load(f)
        self._materialization_cache[op.uuid] = result
        return result

    def write_arrow(self, op: OpSpec, data: pa.Table | pa.Array) -> bool:
        """Write Arrow table data for an operation."""
        log = logger.bind(op=op.uuid)
        # Auto-wrap arrays in single-column tables
        if isinstance(data, (pa.Array, pa.ChunkedArray)):
            name = str(op.uuid)
            if isinstance(data, pa.ChunkedArray):
                data = data.combine_chunks()
            table = pa.Table.from_arrays([data], names=[name])
        elif isinstance(data, pa.Table):
            table = data
        else:
            raise ValueError(f"Expected pa.Table or pa.Array, got {type(data)}")

        # Always cache the result
        self._materialization_cache[op.uuid] = table

        # Only write to store if not ephemeral
        if op.is_ephemeral:
            return True

        log.debug("write_arrow()")
        with self._open_for_data(op, RESULT_FORMATS["arrow"], "wb") as f:
            pq.write_table(table, f)
        self._finalize_result(op)

        return True

    def write_json(self, op: OpSpec, data: dict) -> bool:
        """Write JSON data for an operation."""
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict, got {type(data)}")

        # Verify parse roundtrip.
        # (Also deals w/ non-serializable types up front.)
        data = json.loads(json.dumps(data))

        # Always cache the result
        self._materialization_cache[op.uuid] = data

        # Only write to store if not ephemeral
        if op.is_ephemeral:
            return True

        log = logger.bind(op=op.uuid)
        log.debug("write_json()")
        with self._open_for_data(op, RESULT_FORMATS["json"], "wt") as f:

            json.dump(data, f)
        self._finalize_result(op)
        return True

    def write_numpy(self, op: OpSpec, data: np.ndarray) -> bool:
        """Write numpy array data for an operation."""
        if not isinstance(data, np.ndarray):
            raise ValueError(f"Expected np.ndarray, got {type(data)}")

        # Convert to Arrow table and store as Arrow
        table = self._numpy_to_arrow_table(data, str(op.uuid))
        return self.write_arrow(op, table)

    def _numpy_to_arrow_table(
        self, x: np.ndarray, name: str, kind: str = "vector"
    ) -> pa.Table:
        """Convert numpy array to Arrow table.

        Matches MaterializedResult.from_numpy() logic:
        - kind="vector":
          * 1d → single scalar column
          * 2d → one FixedSizeList column with list_size = x.shape[1]
        - kind="columns":
          * 2d → one scalar column per input column
        """
        if x.ndim == 1:
            arr = pa.array(x)
            return pa.Table.from_arrays([arr], names=[name])

        if x.ndim == 2:
            if kind == "columns":
                arrays = [pa.array(x[:, j]) for j in range(int(x.shape[1]))]
                names = [f"{name}_{j}" for j in range(int(x.shape[1]))]
                return pa.Table.from_arrays(arrays, names=names)
            # default: vector → FixedSizeList
            flat = pa.array(x.reshape(-1))
            fsl = pa.FixedSizeListArray.from_arrays(flat, list_size=int(x.shape[1]))
            return pa.Table.from_arrays([fsl], names=[name])

        raise ValueError(f"unsupported numpy shape {x.shape}")

    def _column_to_numpy(self, col: pa.ChunkedArray | pa.Array) -> np.ndarray:
        """Convert Arrow column to numpy array. Matches MaterializedResult helper."""
        if isinstance(col, pa.ChunkedArray):
            col = col.combine_chunks()
        if isinstance(col.type, pa.FixedSizeListType):
            d = int(col.type.list_size)
            base = col.values.to_numpy(zero_copy_only=False)
            return base.reshape(-1, d)
        return col.to_numpy(zero_copy_only=False)

    def to_sklearn_estimator(self, op: OpSpec) -> Any:
        from sklearn.base import BaseEstimator  # lazy import for performance

        if op.uuid in self._materialization_cache:
            cached_result = self._materialization_cache[op.uuid]
            if isinstance(cached_result, BaseEstimator):
                return cached_result
            else:
                raise ValueError(
                    "Result type doesn't match expected type for to_sklearn_estimator()"
                )
        if self._materialize_if_needed(op):
            return self.to_sklearn_estimator(op)  # load from cache

        log = logger.bind(op=op.uuid)
        log.debug("Loading sklearn estimator from store")
        with self._open_for_data(op, RESULT_FORMATS["pickle"], "rb") as f:
            model = pickle.load(f)
        self._materialization_cache[op.uuid] = model
        return model

    def write_sklearn_estimator(self, op: OpSpec, data: Any) -> bool:
        self._materialization_cache[op.uuid] = data
        if op.is_ephemeral:
            return True
        log = logger.bind(op=op.uuid)
        log.debug("writing sklearn estimator to store")
        with self._open_for_data(op, RESULT_FORMATS["pickle"], "wb") as f:
            pickle.dump(data, f)
        self._finalize_result(op)

        return True


@LocalArrowRunner.implementation
def load_parquet_dataset(runner, op: LoadLocalParquetDatasetOp):
    with fsspec.open(op.file_path, "rb") as f:
        table = pq.read_table(f)
    runner.write_arrow(op, table)


@LocalArrowRunner.implementation
def select_column(runner, op: SelectColumnOp):
    # TODO: should `op` above be a SelectVectorColumnOp | SelectTextColumnOp | ... ?
    dataset = runner.to_arrow(op.dataset)
    column = dataset[op.column_name]
    runner.write_arrow(op, column)


@LocalArrowRunner.implementation
def assign_train_test_split(runner, op: AssignTrainTestSplitOp):
    def _normalize_size(name: str, value: float | int, total: int) -> int:
        if isinstance(value, float):
            if not 0 < value < 1:
                raise ValueError(
                    f"{name}_size as float must be in the open interval (0, 1); got {value}."
                )
            scaled = math.ceil(value * total) if name == "test" else math.floor(value * total)
            return min(scaled, total)
        if isinstance(value, int):
            if value < 0 or value > total:
                raise ValueError(
                    f"{name}_size int must be between 0 and the dataset length ({total}); got {value}."
                )
            return value
        raise TypeError(f"Unsupported type for {name}_size: {type(value)}")

    def _resolve_split_counts(total_rows: int) -> tuple[int, int]:
        if total_rows == 0:
            return 0, 0

        test_specified = op.test_size is not None
        train_specified = op.train_size is not None

        if not test_specified and not train_specified:
            n_test = math.ceil(0.25 * total_rows)
            n_train = total_rows - n_test
            return n_train, n_test

        n_test = _normalize_size("test", op.test_size, total_rows) if test_specified else None
        n_train = (
            _normalize_size("train", op.train_size, total_rows) if train_specified else None
        )

        if n_test is None:
            if n_train is None:
                raise RuntimeError("Unexpected missing split sizes")
            n_test = total_rows - n_train
        elif n_train is None:
            n_train = total_rows - n_test

        if n_test < 0 or n_train < 0:
            raise ValueError(
                f"train/test sizes produced negative allocations (train={n_train}, test={n_test})."
            )

        if n_train + n_test != total_rows:
            if test_specified and train_specified:
                raise ValueError(
                    f"train_size ({n_train}) + test_size ({n_test}) must equal dataset size ({total_rows})."
                )
            # Adjust complement to cover all rows (guards against float rounding issues).
            n_train = total_rows - n_test

        if n_test > total_rows or n_train > total_rows:
            raise ValueError(
                f"train/test sizes cannot exceed dataset size ({total_rows}); got train={n_train}, test={n_test}."
            )

        return n_train, n_test

    table = runner.to_arrow(op.dataset)
    total_rows = len(table)

    n_train, n_test = _resolve_split_counts(total_rows)

    if total_rows == 0:
        runner.write_arrow(op, pa.array([], type=pa.string()))
        return

    rng = random.Random(op.random_state)
    test_indices = set(rng.sample(range(total_rows), n_test)) if n_test else set()

    assignments = ["train"] * total_rows
    for idx in test_indices:
        assignments[idx] = "test"

    if assignments.count("test") != n_test or assignments.count("train") != n_train:
        raise RuntimeError("Split assignment mismatch")

    result = pa.array(assignments, type=pa.string())
    runner.write_arrow(op, result)


@LocalArrowRunner.implementation
def take_rows(runner, op: TakeRowsOp):
    table = runner.to_arrow(op.dataset)
    table = table[op.offset :: op.skip]
    if op.num_rows is not None:
        table = table[: op.num_rows]
    runner.write_arrow(op, table)


@LocalArrowRunner.implementation
def make_umap_viz(runner, op: UMAPVizOp):
    log = logger.bind(op=op.uuid)
    import umap

    dataset = runner.to_numpy(op.input_embedding).astype(np.float32)
    kwds = op.model_dump()
    del kwds["type"]
    del kwds["input_embedding"]
    reducer = umap.UMAP(verbose=True, **kwds)
    log.debug("Running UMAP", **kwds)
    embedding = reducer.fit_transform(dataset)
    log.debug("UMAP completed", shape=embedding.shape)
    runner.write_numpy(op, embedding)


@LocalArrowRunner.implementation
def registry_get_layer_activations(runner, op: LLMLayerActivationsOp):
    return get_layer_activations(runner, op)

@LocalArrowRunner.implementation
def registry_get_llm_output_logits(runner, op: LLMLogitScoresOp):
    return get_llm_output_logits(runner, op)


@LocalArrowRunner.implementation
def from_list_dataset(runner, op: LoadInlineJsonDatasetOp):
    """Convert Python list data to Arrow table."""
    table = pa.table(op.data)
    runner.write_arrow(op, table)


@LocalArrowRunner.implementation
def grouped_op(runner, op: GroupedOp):
    """Run a GroupedOp by running each op in sequence and returning the last result."""
    result = None
    for sub_op in op.ops:
        runner._materialize_if_needed(sub_op)
        result = runner.to_arrow(sub_op)
    # Store the final result for the GroupedOp
    if result is not None:
        runner.write_arrow(op, result)


@LocalArrowRunner.implementation
def vector_to_scalar(runner, op: VectorToScalarOp):
    """Extract a scalar column from a vector column at a given index."""
    vector_result = runner.to_arrow(op.input)

    if isinstance(vector_result, pa.Table):
        # reminder: vectors are stored as FixedSizeList in a single column
        if vector_result.num_columns != 1:
            raise ValueError(
                f"Vector input should have exactly one column; got {vector_result.num_columns}."
            )
        vector_column = vector_result.column(0)
    else:
        vector_column = vector_result

    if isinstance(vector_column, pa.ChunkedArray):
        vector_column = vector_column.combine_chunks()

    if op.col_index < 0:
        raise ValueError(f"col_index must be non-negative; got {op.col_index}.")

    if isinstance(vector_column.type, pa.FixedSizeListType):
        list_size = int(vector_column.type.list_size)
        if op.col_index >= list_size:
            raise ValueError(
                f"col_index {op.col_index} out of bounds for vector length {list_size}."
            )
    elif isinstance(vector_column.type, pa.ListType):
        if len(vector_column) > 0:
            lengths = pc.list_value_length(vector_column)
            if isinstance(lengths, pa.ChunkedArray):
                lengths = lengths.combine_chunks()
            min_length_scalar = pc.min(
                lengths, options=pc.ScalarAggregateOptions(skip_nulls=True)
            )
            min_length = min_length_scalar.as_py() if min_length_scalar is not None else None
            if min_length is not None and op.col_index >= min_length:
                raise ValueError(
                    f"col_index {op.col_index} out of bounds for shortest vector length {min_length}."
                )
    else:
        raise TypeError(
            f"VectorToScalarOp expects a list-like column; got {vector_column.type}."
        )

    scalar_array = pc.list_element(vector_column, op.col_index)
    runner.write_arrow(op, scalar_array)


@LocalArrowRunner.implementation
def pairwise_arithmetic(runner, op: PairwiseArithmeticOp):
    """Apply pairwise arithmetic operations to score columns."""
    left_np = runner.to_numpy(op.left)
    right_np = runner.to_numpy(op.right)

    if left_np.shape != right_np.shape:
        raise ValueError(
            f"Score column shapes must match; got {left_np.shape} and {right_np.shape}."
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        if op.operation == "+":
            result_np = left_np + right_np
        elif op.operation == "-":
            result_np = left_np - right_np
        elif op.operation == "*":
            result_np = left_np * right_np
        elif op.operation == "/":
            result_np = left_np / right_np
        else:
            raise ValueError(f"Unsupported operation: {op.operation}")

    result_array = pa.array(result_np)
    runner.write_arrow(op, result_array)


@LocalArrowRunner.implementation
def category_to_boolean(runner, op: CategoryToBooleanOp):
    """Convert a categorical column to a boolean column."""
    category_result = runner.to_arrow(op.input_category)

    if len(category_result) == 0:
        result = pa.array([], type=pa.bool_())
        runner.write_arrow(op, result)
        return

    if isinstance(category_result, pa.Table):
        category_col = category_result.column(0)
    else:
        category_col = category_result

    if op.true_values is None and op.false_values is None:
        raise ValueError(
            "At least one of true_values or false_values must be provided."
        )

    if op.true_values is not None:
        if op.true_values == []:
            raise ValueError("true_values list is empty.")
        true_values = pa.array(op.true_values)
        if op.false_values is not None:
            if op.false_values == []:
                raise ValueError("false_values list is empty.")
            expected_values = set(op.true_values) | set(op.false_values)
            observed_values = set(category_col.to_pylist())
            if not observed_values.issubset(expected_values):
                raise ValueError(
                    f"The set of actual values in the category column, {observed_values}, must be a subset "
                    f"of true_values.union(false_values), {expected_values}."
                )

        boolean_array = pc.is_in(category_col, true_values)
        runner.write_arrow(op, boolean_array)
    else:
        if op.false_values == []:
            raise ValueError("false_values list is empty.")
        # no true values, but false values are specified
        false_values = pa.array(op.false_values)
        boolean_array = pc.invert(pc.is_in(category_col, false_values))
        runner.write_arrow(op, boolean_array)


@LocalArrowRunner.implementation
def mask_rows(runner, op: MaskRowsOp):
    """Filter rows in the dataset based on a boolean mask."""
    log = logger.bind(op=op.uuid)
    dataset_table = runner.to_arrow(op.dataset)
    mask_result = runner.to_arrow(op.mask)
    if isinstance(mask_result, pa.Table):
        boolean_array = mask_result.column(0)
    else:
        boolean_array = mask_result

    # Handle empty datasets - if there are no rows, return the empty table directly
    if len(boolean_array) == 0:
        runner.write_arrow(op, dataset_table)
        return

    ## Ensure the boolean array has the correct type for filtering
    # if boolean_array.type != pa.bool_():
    #    boolean_array = pc.cast(boolean_array, pa.bool_())

    if len(boolean_array) != len(dataset_table):
        raise ValueError("Mask length must match dataset row count")
    log.debug(
        "Applying mask filter",
        dataset_rows=len(dataset_table),
        true_count=pc.sum(boolean_array).as_py(),
    )

    filtered_table = pc.filter(dataset_table, boolean_array)
    runner.write_arrow(op, filtered_table)


@LocalArrowRunner.implementation
def boolean_op(runner, op: BooleanLogicOp):
    """Perform a boolean operation on two columns."""
    left_result = runner.to_arrow(op.left)
    right_result = runner.to_arrow(op.right)
    if len(left_result) != len(right_result):
        raise ValueError("Both columns must have the same length.")
    if len(left_result) == 0 or len(right_result) == 0:
        result = pa.array([], type=pa.bool_())
        runner.write_arrow(op, result)
        return

    if isinstance(left_result, pa.Table):
        left = left_result.column(0)
    else:
        left = left_result
    if isinstance(right_result, pa.Table):
        right = right_result.column(0)
    else:
        right = right_result

    if left.type != pa.bool_() or right.type != pa.bool_():
        raise ValueError("Both columns must be boolean.")

    if op.operation == "and":
        result = pc.and_(left, right)
    elif op.operation == "or":
        result = pc.or_(left, right)
    elif op.operation == "xor":
        result = pc.xor(left, right)
    elif op.operation == "not":
        result = pc.invert(left)
    else:
        raise ValueError(f"Unknown operator: {op.operation}")

    runner.write_arrow(op, result)



@LocalArrowRunner.implementation
def jinja_templatize(runner, op: JinjaTemplatizeOp):
    """Apply Jinja2 template with context from text columns."""
    import jinja2

    log = logger.bind(op=op.uuid)
    log.debug("Running Jinja templatization", template=op.template[:100])

    # Create Jinja2 environment
    env = jinja2.Environment(autoescape=False)  # noqa: S701, prompts aren't HTML/XML
    template = env.from_string(op.template)

    # Materialize all context columns
    context_data = {}
    for key, text_column in op.context.items():
        column_result = runner.to_arrow(text_column)
        if isinstance(column_result, pa.Table):
            column_result = column_result.column(0)
        context_data[key] = column_result.to_pylist()

    # Determine the length (all columns should have the same length)
    if context_data:
        lengths = [len(values) for values in context_data.values()]
        if not all(length == lengths[0] for length in lengths):
            raise ValueError("All context columns must have the same length")
        num_rows = lengths[0]
    else:
        num_rows = 1  # If no context, generate template once

    # Apply template to each row
    results = []
    for i in range(num_rows):
        # Build context for this row
        row_context = {}
        for key, values in context_data.items():
            row_context[key] = values[i]

        # Render template
        rendered = template.render(**row_context)
        results.append(rendered)

    log.debug("Jinja templatization completed", num_results=len(results))
    result_array = pa.array(results)
    runner.write_arrow(op, result_array)
