# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable, Literal, TypeVar

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from krnel.graph.classifier_ops import ClassifierEvaluationOp
from krnel.graph.op_spec import OpSpec
from krnel.logging import get_logger

logger = get_logger(__name__)

OpSpecT = TypeVar('OpSpecT', bound=OpSpec)
T = TypeVar('T')

_IMPLEMENTATIONS = []

class MaterializedResult:

    @classmethod
    def from_any(cls, data: Any, op: OpSpec) -> MaterializedResult:
        for subclass in cls.__subclasses__():
            try:
                return subclass.from_any(data, op=op)
            except NotImplementedError:
                continue
        raise ValueError(f"Cannot convert {type(data)} to MaterializedResult")

    @classmethod
    def implementation(cls, kls):
        _IMPLEMENTATIONS.append(kls)
        return kls

    def to_numpy(self):
        raise NotImplementedError()
    def to_arrow(self):
        raise NotImplementedError()
    def to_pandas(self):
        raise NotImplementedError()
    def write_to(self, file):
        raise NotImplementedError()

class MaterializedResultTable(MaterializedResult):
    """
    single envelope that always holds a pyarrow.Table.
    a "series" is just a 1-column table.

    invariants:
    - self.table is a pa.Table
    - vector columns are FixedSizeList
    - scalar columns are non-list primitive types
    """

    def __init__(self, table: pa.Table, op: OpSpec | None):
        if not isinstance(table, pa.Table):
            raise TypeError(f"expected pa.Table, got {type(table)}")
        self.table = table
        self.op = op

    @classmethod
    def from_arrow_table(cls, table: pa.Table, op: OpSpec) -> "MaterializedResult":
        return cls(table, op=op)

    @classmethod
    def from_arrow_array(cls, array: pa.Array | pa.ChunkedArray, name: str, op: OpSpec) -> "MaterializedResult":
        if isinstance(array, pa.ChunkedArray):
            array = array.combine_chunks()
        if not isinstance(array, pa.Array):
            raise TypeError(f"expected pa.Array/ChunkedArray, got {type(array)}")
        table = pa.Table.from_arrays([array], names=[name])
        return cls(table, op=op)

    @classmethod
    def from_numpy(
        cls,
        x: np.ndarray,
        name: str,
        op: OpSpec,
        kind: Literal["vector", "columns"] = "vector",
    ) -> "MaterializedResult":
        """
        - kind="vector":
          * 1d → single scalar column
          * 2d → one FixedSizeList column with list_size = x.shape[1]
        - kind="columns":
          * 2d → one scalar column per input column
        """
        if not isinstance(x, np.ndarray):
            raise TypeError(f"expected np.ndarray, got {type(x)}")

        if x.ndim == 1:
            arr = pa.array(x)
            tbl = pa.Table.from_arrays([arr], names=[name])
            return cls(tbl, op=op)

        if x.ndim == 2:
            if kind == "columns":
                arrays = [pa.array(x[:, j]) for j in range(int(x.shape[1]))]
                names = [f"{name}_{j}" for j in range(int(x.shape[1]))]
                return cls(pa.Table.from_arrays(arrays, names=names), op=op)
            # default: vector → FixedSizeList
            flat = pa.array(x.reshape(-1))
            fsl = pa.FixedSizeListArray.from_arrays(flat, list_size=int(x.shape[1]))
            return cls(pa.Table.from_arrays([fsl], names=[name]), op=op)

        raise ValueError(f"unsupported numpy shape {x.shape}")

    @classmethod
    def from_pydict(cls, obj: dict[str, Any], op: OpSpec) -> "MaterializedResult":
        return cls(pa.Table.from_pydict(obj), op=op)

    @classmethod
    def from_any(cls, data: Any, op: OpSpec) -> "MaterializedResult":
        """best-effort ingestion that **always** returns MaterializedResult."""
        if hasattr(data, 'read'):
            try:
                data.seek(0)
                data = pq.read_table(data)
            except pa.ArrowInvalid:
                raise NotImplementedError(f"Cannot read {type(data)} as Arrow table")


        if isinstance(data, MaterializedResult):
            return data
        if isinstance(data, pa.Table):
            return cls.from_arrow_table(data, op=op)
        if isinstance(data, (pa.Array, pa.ChunkedArray)):
            return cls.from_arrow_array(data, name=str(op.uuid), op=op)
        if isinstance(data, dict) and not isinstance(op, ClassifierEvaluationOp):
            return cls.from_pydict(data, op=op)
        if isinstance(data, list) and not isinstance(op, ClassifierEvaluationOp):
            return cls.from_pydict({str(op.uuid): data}, op=op)
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                return cls.from_numpy(data, name=str(op.uuid), kind="vector", op=op)
            if data.ndim == 2 and data.shape[0] != 1:
                return cls.from_numpy(data, name=str(op.uuid), kind="vector", op=op)
            raise ValueError(f"result of {op.uuid} is an unsupported numpy array shape: {data.shape}")
        raise NotImplementedError(f"result of {op.uuid} is not a valid Arrow-compatible object: {type(data)}")

    def to_arrow(self) -> pa.Table:
        return self.table

    def to_pandas(self):
        return self.table.to_pandas()

    def to_numpy(self, name: str | int | None = None) -> np.ndarray:
        """
        - if name provided → extract that column only
        - if single column → return that column
        - if multi-column and all scalar → np.column_stack
        - if multi-column with any FixedSizeList → require a name (ambiguous); raise
        """
        tbl = self.table

        if tbl.num_columns == 1:
            assert name is None
            return _column_to_numpy(tbl.column(0))

        if name is not None:
            col = tbl.column(name) if isinstance(name, str) else tbl.column(int(name))
            return _column_to_numpy(col)

        # multi-col: ensure all scalar
        cols = [c for c in tbl.itercolumns()]
        if any(isinstance(c.type, pa.FixedSizeListType) for c in cols):
            raise ValueError("to_numpy() ambiguous for multi-column table containing vector columns; specify a column name")
        arrays = [c.combine_chunks().to_numpy(zero_copy_only=False) for c in cols]
        return np.column_stack(arrays)

    # ----- optional hfds adapters (lazy imports) -----
    def to_hfds(self):  # pragma: no cover
        try:
            import datasets  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise RuntimeError("huggingface datasets is not installed") from e
        return datasets.Dataset.from_pandas(self.table.to_pandas())

    def write_to(self, file):
        table = self.to_arrow()
        logger.debug("Saving Arrow dataset to file", schema=table.schema, shape=table.shape)
        pq.write_table(table, file)


def _column_to_numpy(col: pa.ChunkedArray | pa.Array) -> np.ndarray:
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    if isinstance(col.type, pa.FixedSizeListType):
        d = int(col.type.list_size)
        base = col.values.to_numpy(zero_copy_only=False)
        return base.reshape(-1, d)
    return col.to_numpy(zero_copy_only=False)


class MaterializedJSONResult(MaterializedResult):
    @classmethod
    def from_any(cls, data: Any, op: OpSpec) -> "MaterializedJSONResult":
        if hasattr(data, 'read') and isinstance(op, ClassifierEvaluationOp):
            data.seek(0)
            try:
                return cls(json.load(data), op=op)
            except json.JSONDecodeError:
                raise NotImplementedError(f"Cannot read {type(data)} as JSON")
        if isinstance(data, dict):
            return cls(data, op=op)
        raise NotImplementedError(f"Cannot read {type(data)} as JSON")

    def __init__(self, data: dict, op: OpSpec | None):
        if not isinstance(data, dict):
            raise TypeError(f"expected dict, got {type(data)}")
        self.data = data
        self.op = op

    def write_to(self, file):
        logger.debug("Saving JSON to file")
        buffer = json.dumps(self.data)
        file.write(buffer.encode('utf-8'))