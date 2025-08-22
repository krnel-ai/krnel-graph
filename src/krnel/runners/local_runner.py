# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Annotated
import warnings

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
from krnel.runners.materialized_result import MaterializedResult
from krnel.runners.model_registry import get_layer_activations

logger = get_logger(__name__)

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
                        self.materialize(dataset)
                self._materialized_datasets.add(dataset.uuid)



    def from_list(self, data: dict[str, list[Any]]) -> FromListOp:
        """Create a FromListOp from Python lists/dicts."""
        return FromListOp(
            content_hash=sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(),
            data=data,
        )

    def get_result(self, spec: OpSpec) -> pa.Table:
        if spec.uuid in self._materialization_cache:
            log = logger.bind(op=spec.uuid, cached=True)
            log.debug("get_result() - using cached result")
            return self._materialization_cache[spec.uuid]
        path = self._path(spec, _RESULT_PQ_FILE_SUFFIX)
        log = logger.bind(op=spec.uuid, path=path, cached=False)
        log.debug("get_result()")
        with self.fs.open(path, "rb") as f:
            result = MaterializedResult.from_any(f, spec)
        self._materialization_cache[spec.uuid] = result
        return result

    def has_result(self, spec: OpSpec) -> bool:
        path = self._path(spec, _RESULT_PQ_FILE_SUFFIX)
        log = logger.bind(op=spec.uuid, path=path)
        result = self.fs.exists(path)
        log.debug("has_result()", result=result)
        return result

    def put_result(self, spec: OpSpec, result: Any) -> bool:
        if spec.is_ephemeral:
            return True
        path = self._path(spec, _RESULT_PQ_FILE_SUFFIX)
        log = logger.bind(op=spec.uuid, path=path)
        log.debug("put_result()")
        with self.fs.open(path, "wb") as f:
            return result.write_to(f)
        return True

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
            log.debug("get_status() - not found, creating new")
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


@LocalArrowRunner.implementation
def load_parquet_dataset(runner, op: LoadLocalParquetDatasetOp):
    with fsspec.open(op.file_path, "rb") as f:
        return pq.read_table(f)


@LocalArrowRunner.implementation
def select_column(runner, op: SelectColumnOp):
    # TODO: should `op` above be a SelectVectorColumnOp | SelectTextColumnOp | ... ?
    dataset = runner.materialize(op.dataset).to_arrow()
    return dataset[op.column_name]

@LocalArrowRunner.implementation
def take_rows(runner, op: TakeRowsOp):
    table = runner.materialize(op.dataset).to_arrow()
    table = table[op.offset::op.skip]
    if op.num_rows is not None:
        return table[:op.num_rows]
    return table


@LocalArrowRunner.implementation
def make_umap_viz(runner, op: UMAPVizOp):
    log = logger.bind(op=op.uuid)
    import umap
    dataset = runner.materialize(op.input_embedding).to_numpy().astype(np.float32)
    kwds = op.model_dump()
    del kwds['type']
    del kwds['input_embedding']
    reducer = umap.UMAP(verbose=True, **kwds)
    log.debug("Running UMAP", **kwds)
    embedding = reducer.fit_transform(dataset)
    log.debug("UMAP completed", shape=embedding.shape)
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


@LocalArrowRunner.implementation
def category_to_boolean(runner, op: CategoryToBooleanOp):
    """Convert a categorical column to a boolean column."""
    category_col = runner.materialize(op.input_category).to_arrow()

    if len(category_col) == 0:
        return pa.array([], type=pa.bool_())
    if isinstance(category_col, pa.Table):
        category_col = category_col.column(0)
    else:
        category_col = category_col

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
        return boolean_array
    else:
        if op.false_values == []:
            raise ValueError("false_values list is empty.")
        # no true values, but false values are specified
        false_values = pa.array(op.false_values)
        return pc.invert(pc.is_in(category_col, false_values))


@LocalArrowRunner.implementation
def mask_rows(runner, op: MaskRowsOp):
    """Filter rows in the dataset based on a boolean mask."""
    log = logger.bind(op=op.uuid)
    dataset_table = runner.materialize(op.dataset).to_arrow()
    mask_column = runner.materialize(op.mask).to_arrow()
    if isinstance(mask_column, pa.Table):
        boolean_array = mask_column.column(0)
    else:
        boolean_array = mask_column

    # Handle empty datasets - if there are no rows, return the empty table directly
    if len(boolean_array) == 0:
        return dataset_table

    ## Ensure the boolean array has the correct type for filtering
    #if boolean_array.type != pa.bool_():
    #    boolean_array = pc.cast(boolean_array, pa.bool_())

    assert len(boolean_array) == len(dataset_table), "Mask length must match dataset row count"
    log.debug("Applying mask filter",
              dataset_rows=len(dataset_table),
              true_count=pc.sum(boolean_array).as_py())

    filtered_table = pc.filter(dataset_table, boolean_array)

    return filtered_table

@LocalArrowRunner.implementation
def boolean_op(runner, op: BooleanLogicOp):
    """Perform a boolean operation on two columns."""
    left = runner.materialize(op.left).to_arrow()
    right = runner.materialize(op.right).to_arrow()
    if len(left) != len(right):
        raise ValueError("Both columns must have the same length.")
    if len(left) == 0 or len(right) == 0:
        return pa.array([], type=pa.bool_())
    if isinstance(left, pa.Table):
        left = left.column(0)
    if isinstance(right, pa.Table):
        right = right.column(0)

    if left.type != pa.bool_() or right.type != pa.bool_():
        raise ValueError("Both columns must be boolean.")

    if op.operation == "and":
        return pc.and_(left, right)
    elif op.operation == "or":
        return pc.or_(left, right)
    elif op.operation == "xor":
        return pc.xor(left, right)
    elif op.operation == "not":
        return pc.invert(left)
    else:
        raise ValueError(f"Unknown operator: {op.operation}")


@LocalArrowRunner.implementation
def evaluate_scores(runner, op: ClassifierEvaluationOp):
    """Evaluate classification scores."""
    from sklearn import metrics
    log = logger.bind(op=op.uuid)
    y_true = runner.materialize(op.y_groundtruth).to_numpy()
    y_score = runner.materialize(op.y_score).to_numpy()
    splits = runner.materialize(op.split).to_numpy()
    domain = runner.materialize(op.predict_domain).to_numpy()

    per_split_metrics = defaultdict(dict)
    def compute_classification_metrics(y_true, y_score):
        """Appropriate for binary classification results."""
        result = {}
        result[f"count"] = len(y_true)
        result[f"n_true"] = int(y_true.sum())
        prec, rec, thresh = metrics.precision_recall_curve(y_true, y_score)
        #result[f"pr_curve"] = {
        #    "precision": prec.tolist(),
        #    "recall": rec.tolist(),
        #    "threshold": thresh.tolist(),
        #}
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

    for split in set(splits):
        split_mask = (splits == split) & domain
        per_split_metrics[split] = compute_classification_metrics(y_true[split_mask], y_score[split_mask])
    log.error("Metrics are here", **per_split_metrics)
    return per_split_metrics


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
        column_result = runner.materialize(text_column).to_arrow()
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
    return pa.array(results)
