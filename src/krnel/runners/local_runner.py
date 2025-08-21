# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Annotated
import warnings
import pickle

from collections import defaultdict

from krnel.graph import SelectColumnOp
from krnel.graph.classifier_ops import ClassifierEvaluationOp, TrainClassifierOp
from krnel.graph.dataset_ops import BooleanLogicOp, CategoryToBooleanOp, LoadDatasetOp, SelectCategoricalColumnOp, SelectScoreColumnOp, SelectVectorColumnOp, SelectTextColumnOp, SelectTrainTestSplitColumnOp, TakeRowsOp, FromListOp, MaskRowsOp, JinjaTemplatizeOp
from krnel.graph.llm_ops import LLMLayerActivationsOp
from krnel.graph.op_spec import OpSpec, graph_deserialize, graph_serialize, ExcludeFromUUID
from krnel.graph.grouped_ops import GroupedOp
from krnel.graph.types import DatasetType
from krnel.graph.viz_ops import UMAPVizOp
from krnel.logging import get_logger
from krnel.runners.base_runner import BaseRunner

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import fsspec

from krnel.runners.op_status import OpStatus
from krnel.runners.model_registry import get_layer_activations

logger = get_logger(__name__)

# Global dictionary for result file formats
_RESULT_FORMATS = {
    "arrow": "result.parquet",
    "json": "result.json",
    "pickle": "result.pickle"
}
_RESULT_INDICATOR = "done"
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
        """Initialize runner with an fsspec filesystem and a base path within it.

        - if only root_path is provided (e.g., "s3://bucket/prefix" or "/tmp/krnel"), infer fs via fsspec.
        - if filesystem is provided, root_path should be a path valid for that fs (protocol will be stripped if present).
        - defaults to in-memory fs when nothing given.
        """
        self._materialization_cache = {}
        self._store_uri = store_uri
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
            classname, uuid_hash_only = OpSpec.parse_uuid(spec)
        else:
            classname = spec.__class__.__name__
            uuid_hash_only = spec.uuid_hash
        dir_path = self._join(
            self.store_path_base,
            classname,
            uuid_hash_only,
        )
        file_path = self._join(dir_path, extension)
        self.fs.makedirs(dir_path, exist_ok=True)
        return file_path

    def from_parquet(self, path: str) -> LoadLocalParquetDatasetOp:
        """Create a LoadParquetDatasetOp from a Parquet file path (local or remote)."""
        # compute content hash by streaming bytes; fsspec.open infers the fs from the URL
        log = logger.bind(path=path)
        h = sha256()
        log.debug("Reading parquet dataset")
        with fsspec.open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        log.debug("Content hash", content_hash=h.hexdigest())
        return LoadLocalParquetDatasetOp(
            content_hash=h.hexdigest(),
            file_path=path,
        )

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
                        log.debug("prepare(): dataset needs materializing", dataset=dataset)
                        self._materialize_if_needed(dataset)
                self._materialized_datasets.add(dataset.uuid)



    def from_list(self, data: dict[str, list[Any]]) -> FromListOp:
        """Create a FromListOp from Python lists/dicts."""
        return FromListOp(
            content_hash=sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(),
            data=data,
        )


    def has_result(self, spec: OpSpec) -> bool:
        if spec.is_ephemeral:
            return True  # Ephemeral ops are always "available"

        # Check if any result format exists
        log = logger.bind(op=spec.uuid)
        done_indicator = self._path(spec, _RESULT_INDICATOR)
        if self.fs.exists(done_indicator):
            log.debug("has_result()", result=True, path=done_indicator)
            return True
        #for format_name, suffix in _RESULT_FORMATS.items():
        #    path = self._path(spec, suffix)
        #    if self.fs.exists(path):
        #        log.debug("has_result()", result=True, format=format_name, path=path)
        #        return True

        log.debug("has_result()", result=False)
        return False

    def uuid_to_op(self, uuid: str) -> OpSpec | None:
        log = logger.bind(uuid=uuid)
        path = self._path(uuid, _STATUS_JSON_FILE_SUFFIX)
        if self.fs.exists(path):
            log.debug("uuid_to_op()", exists=True)
            with self.fs.open(path, "rt") as f:
                text = f.read()
            result = json.loads(text)
            results = graph_deserialize(result['op'])
            return results[0]
        log.debug("uuid_to_op()", exists=False)
        return None

    def get_status(self, spec: OpSpec) -> OpStatus:
        if spec.is_ephemeral:
            # Ephemeral ops do not have a status file, they are always 'ephemeral'
            return OpStatus(op=spec, state='ephemeral')
        path = self._path(spec, _STATUS_JSON_FILE_SUFFIX)
        log = logger.bind(op=spec.uuid, path=path)
        log.debug("get_status()")
        if self.fs.exists(path):
            with self.fs.open(path, "rt") as f:
                result = json.load(f)
            # Need to deserialize OpSpec separately
            [result['op']] = graph_deserialize(result['op'])
            status = OpStatus.model_validate(result)
            return status
        else:
            log.debug("status not found, creating new")
            new_status = OpStatus(op=spec, state='new')
            self.put_status(new_status)
            return new_status

    def put_status(self, status: OpStatus) -> bool:
        if status.op.is_ephemeral:
            # Ephemeral ops do not have a status file, they are always 'ephemeral'
            return True
        path = self._path(status.op, _STATUS_JSON_FILE_SUFFIX)
        log = logger.bind(op=status.op.uuid, path=path)
        log.debug("put_status()", state=status.state)
        with self.fs.open(path, "wt") as f:
            f.write(status.model_dump_json())
        return True

    # Implementation of BaseRunner abstract methods
    def to_arrow(self, op: OpSpec) -> pa.Table:
        if op.uuid in self._materialization_cache:
            cached_result = self._materialization_cache[op.uuid]
            if isinstance(cached_result, pa.Table):
                return cached_result
            else:
                raise ValueError(f"Result type doesn't match expected type for to_arrow()")

        if self._materialize_if_needed(op):
            return self.to_arrow(op) # load from cache

        path = self._path(op, _RESULT_FORMATS["arrow"])
        log = logger.bind(op=op.uuid, path=path)
        log.debug("Loading arrow result from store")
        with self.fs.open(path, "rb") as f:
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
            raise ValueError(f"to_numpy() expects single-column tables, got {table.num_columns} columns from {type(op).__name__}")

    def to_json(self, op: OpSpec) -> dict:
        if op.uuid in self._materialization_cache:
            cached_result = self._materialization_cache[op.uuid]
            if isinstance(cached_result, dict):
                return cached_result
            else:
                raise ValueError(f"Result type doesn't match expected type for to_json()")

        if self._materialize_if_needed(op):
            return self.to_json(op) # load from cache

        path = self._path(op, _RESULT_FORMATS["json"])
        with self.fs.open(path, "rt") as f:
            import json
            result = json.load(f)
        self._materialization_cache[op.uuid] = result
        return result

    def _finalize_result(self, op: OpSpec):
        done_path = self._path(op, _RESULT_INDICATOR)
        with self.fs.open(done_path, "wt") as f:
            f.write("done")

    def write_arrow(self, op: OpSpec, data: pa.Table | pa.Array) -> bool:
        """Write Arrow table data for an operation."""
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

        path = self._path(op, _RESULT_FORMATS["arrow"])
        log = logger.bind(op=op.uuid, path=path)
        log.debug("write_arrow()")
        with self.fs.open(path, "wb") as f:
            pq.write_table(table, f)
        self._finalize_result(op)

        return True

    def write_json(self, op: OpSpec, data: dict) -> bool:
        """Write JSON data for an operation."""
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict, got {type(data)}")

        # Always cache the result
        self._materialization_cache[op.uuid] = data

        # Only write to store if not ephemeral
        if op.is_ephemeral:
            return True

        path = self._path(op, _RESULT_FORMATS["json"])
        log = logger.bind(op=op.uuid, path=path)
        log.debug("write_json()")
        with self.fs.open(path, "wt") as f:
            import json
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

    def _numpy_to_arrow_table(self, x: np.ndarray, name: str, kind: str = "vector") -> pa.Table:
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
        from sklearn.base import BaseEstimator # lazy import for performance
        if op.uuid in self._materialization_cache:
            cached_result = self._materialization_cache[op.uuid]
            if isinstance(cached_result, BaseEstimator):
                return cached_result
            else:
                raise ValueError(f"Result type doesn't match expected type for to_sklearn_estimator()")
        if self._materialize_if_needed(op):
            return self.to_sklearn_estimator(op) # load from cache

        path = self._path(op, _RESULT_FORMATS["pickle"])
        log = logger.bind(op=op.uuid, path=path)
        log.debug("Loading sklearn estimator from store")
        with self.fs.open(path, "rb") as f:
            model = pickle.load(f)
        self._materialization_cache[op.uuid] = model
        return model

    def write_sklearn_estimator(self, op: OpSpec, data: Any) -> bool:
        self._materialization_cache[op.uuid] = data
        if op.is_ephemeral:
            return True
        path = self._path(op, _RESULT_FORMATS["pickle"])
        log = logger.bind(op=op.uuid, path=path)
        log.debug("writing sklearn estimator to store")
        with self.fs.open(path, "wb") as f:
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
def take_rows(runner, op: TakeRowsOp):
    table = runner.to_arrow(op.dataset)
    table = table[op.offset::op.skip]
    if op.num_rows is not None:
        table = table[:op.num_rows]
    runner.write_arrow(op, table)


@LocalArrowRunner.implementation
def make_umap_viz(runner, op: UMAPVizOp):
    log = logger.bind(op=op.uuid)
    import umap
    dataset = runner.to_numpy(op.input_embedding).astype(np.float32)
    kwds = op.model_dump()
    del kwds['type']
    del kwds['input_embedding']
    reducer = umap.UMAP(verbose=True, **kwds)
    log.debug("Running UMAP", **kwds)
    embedding = reducer.fit_transform(dataset)
    log.debug("UMAP completed", shape=embedding.shape)
    runner.write_numpy(op, embedding)


@LocalArrowRunner.implementation
def registry_get_layer_activations(runner, op: LLMLayerActivationsOp):
    """LLM embedding using the model registry for dispatching."""
    # Use model registry to dispatch based on model_name URL
    return get_layer_activations(runner, op)


@LocalArrowRunner.implementation
def from_list_dataset(runner, op: FromListOp):
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
        raise ValueError("At least one of true_values or false_values must be provided.")

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
    #if boolean_array.type != pa.bool_():
    #    boolean_array = pc.cast(boolean_array, pa.bool_())

    assert len(boolean_array) == len(dataset_table), "Mask length must match dataset row count"
    log.debug("Applying mask filter",
              dataset_rows=len(dataset_table),
              true_count=pc.sum(boolean_array).as_py())

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
def evaluate_scores(runner, op: ClassifierEvaluationOp):
    """Evaluate classification scores."""
    from sklearn import metrics
    log = logger.bind(op=op.uuid)
    scores = runner.to_numpy(op.score)

    gt_positives = runner.to_numpy(op.gt_positives)
    assert gt_positives.dtype == np.bool_
    gt_negatives = runner.to_numpy(op.gt_negatives)
    assert gt_negatives.dtype == np.bool_
    if (n_inconsistent := (gt_positives & gt_negatives).sum()) > 0:
        raise ValueError(f"Some examples ({n_inconsistent}) are both positive and negative")

    per_split_metrics = defaultdict(dict)
    def compute_classification_metrics(y_true, y_score):
        """Appropriate for binary classification results."""
        result = {}
        result[f"count"] = len(y_true)
        result[f"n_true"] = int(y_true.sum())
        prec, rec, thresh = metrics.precision_recall_curve(y_true, y_score)
        # result[f"pr_curve"] = {
        #    "precision": prec.tolist(),
        #    "recall": rec.tolist(),
        #    "threshold": thresh.tolist(),
        # }
        roc_fpr, roc_tpr, roc_thresh = metrics.roc_curve(y_true, y_score)
        # result["roc_curve"] = metrics.roc_curve(y_true, y_score)
        result[f"average_precision"] = metrics.average_precision_score(y_true, y_score)
        result[f"roc_auc"] = metrics.roc_auc_score(y_true, y_score)

        for recall in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]:
            precision = prec[rec >= recall].max()
            if np.isnan(precision):
                precision = 0.0
            result[f"precision@{recall}"] = precision
        return result

    splits = None
    if op.split is not None:
        splits = runner.to_numpy(op.split)

    if splits is None:
        log.debug("No splits provided, grouping all samples into one 'all' split")
        splits = np.array(['all'] * len(scores))

    domain = None
    if op.predict_domain is not None:
        domain = runner.to_numpy(op.predict_domain)
        assert domain.dtype == np.bool_

    if domain is None:
        log.debug("No domain provided, using all samples")
        domain = np.array([True] * len(scores))

    for split in set(splits):
        split_mask = (splits == split) & domain & (gt_positives | gt_negatives)
        per_split_metrics[split] = compute_classification_metrics(
            gt_positives[split_mask], scores[split_mask]
        )

    runner.write_json(op, per_split_metrics)


@LocalArrowRunner.implementation
def jinja_templatize(runner, op: JinjaTemplatizeOp):
    """Apply Jinja2 template with context from text columns."""
    import jinja2

    log = logger.bind(op=op.uuid)
    log.debug("Running Jinja templatization", template=op.template[:100])

    # Create Jinja2 environment
    env = jinja2.Environment()
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
