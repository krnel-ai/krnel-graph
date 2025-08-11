# Copyright (c) 2025 Krnel
# Points of Contact: 
#   - kimmy@krnel.ai

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pyarrow as pa

from krnel.graph.op_spec import OpSpec


class MaterializedResult:
    """
    single envelope that always holds a pyarrow.Table.
    a "series" is just a 1-column table.

    invariants:
    - self.table is a pa.Table
    - vector columns are FixedSizeList
    - scalar columns are non-list primitive types
    """
    op: OpSpec | None

    table: pa.Table

    # ----- construction -----
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
            return cls(tbl)

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
    def from_pydict(cls, obj: dict[str, Any]) -> "MaterializedResult":
        return cls(pa.Table.from_pydict(obj), op=None)

    @classmethod
    def from_any(cls, data: Any, op: OpSpec) -> "MaterializedResult":
        """best-effort ingestion that **always** returns MaterializedResult."""
        if isinstance(data, MaterializedResult):
            return data
        if isinstance(data, pa.Table):
            return cls.from_arrow_table(data, op=op)
        if isinstance(data, (pa.Array, pa.ChunkedArray)):
            return cls.from_arrow_array(data, name=str(op.uuid), op=op)
        if isinstance(data, dict):
            return cls.from_pydict(data, op=op)
        if isinstance(data, list):
            return cls.from_pydict({str(op.uuid): data}, op=op)
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                return cls.from_numpy(data, name=str(op.uuid), kind="vector", op=op)
            if data.ndim == 2 and data.shape[0] != 1:
                return cls.from_numpy(data, name=str(op.uuid), kind="vector", op=op)
            raise ValueError(f"result of {op.uuid} is an unsupported numpy array shape: {data.shape}")
        raise ValueError(f"result of {op.uuid} is not a valid Arrow-compatible object: {type(data)}")

    # ----- conversions -----
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

    @classmethod
    def from_hfds(cls, ds):
        import datasets
        if not isinstance(ds, datasets.Dataset):
            raise TypeError("expected a datasets.Dataset")
        return cls(pa.Table.from_pandas(ds.to_pandas()), op=None)



def _column_to_numpy(col: pa.ChunkedArray | pa.Array) -> np.ndarray:
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    if isinstance(col.type, pa.FixedSizeListType):
        d = int(col.type.list_size)
        base = col.values.to_numpy(zero_copy_only=False)
        return base.reshape(-1, d)
    return col.to_numpy(zero_copy_only=False)