# Copyright (c) 2025-2026 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

import contextlib
import io
import json
import os.path
from uuid import uuid4

import fsspec
import fsspec.implementations.cached
from fsspec.utils import atomic_write

from krnel.graph import config
from krnel.graph.op_spec import OpSpec, graph_deserialize
from krnel.graph.runners.base_runner import BaseRunner
from krnel.graph.runners.local_runner.local_arrow_runner import (
    RESULT_INDICATOR,
    STATUS_JSON_FILE_SUFFIX,
    LocalArrowRunner,
)
from krnel.graph.runners.op_status import OpStatus
from krnel.logging import get_logger

logger = get_logger(__name__)


@contextlib.contextmanager
def cached_open(local_cache_path, mode, remote_open_fun):
    log = logger.bind(local_cache_path=local_cache_path, mode=mode)

    if "w" in mode:
        with atomic_write(local_cache_path, mode) as local_f:
            yield local_f  # client will write here
        log.debug("cache write: copy result to remote")
        with open(local_cache_path, "rb") as local_f:
            with remote_open_fun("wb") as remote_f:
                remote_f.write(local_f.read())
    elif "r" in mode:
        if os.path.exists(local_cache_path):
            log.debug("cache read", is_cache_hit=True)
        else:
            log.debug("cache read", is_cache_hit=False)
            with remote_open_fun("rb") as remote_f:
                with atomic_write(local_cache_path, "wb") as local_f:
                    local_f.write(remote_f.read())
        with open(local_cache_path, mode) as local_f:
            yield local_f
    else:
        raise ValueError(f"unsupported file mode: {mode}")


class LocalCachedRunner(LocalArrowRunner):
    """
    A LocalArrowRunner that's backed by a caching store.

    To read data files (*.parquet, results.json, done, etc):
    - Serve the file from local cache
    - If it doesn't exist, fetch it into cache and serve it

    To read status files (status.json):
    - Serve the status from local cache
    - If it doesn't exist, fetch the status. If status is 'complete', store in cache.

    To write data files (*.parquet) and status files:
    - Write into local cache first
    - Then copy remote

    """

    def __init__(
        self,
        store_uri: str | None = None,
        filesystem: fsspec.AbstractFileSystem | str | None = None,
        cache_path: str | None = None,
        # expiry_time: int = 24 * 60 * 60 * 7,
    ):
        """A runner that's backed by a local cache directory.

        Arguments:
        - store_uri: The URI of the data store (usually remote).
        - filesystem: The filesystem to use for reading/writing data remotely. (Optional, can be parsed from store_uri)
        - cache_path: Location to store cache files. Will default to `tempfile.gettempdir()`
        """
        if cache_path is None:
            self.cache_path = config.KrnelGraphConfig().cache_path
        else:
            self.cache_path = cache_path
        super().__init__(store_uri=store_uri, filesystem=filesystem)

    def _path_in_cache(self, op: OpSpec, basename: str) -> str:
        """Return the cache path of the file.

        Note: Use `open()` or `atomic_write()` on these files,
        not `self.fs.open()` -- these are not remote.
        """
        local_cache_path = self._path(
            op, basename, store_path_base=self.cache_path, makedirs=False
        )
        os.makedirs(os.path.dirname(local_cache_path), exist_ok=True)
        return local_cache_path

    def _open_for_data(self, op: OpSpec, basename: str, mode: str) -> io.IOBase:
        local_cache_path = self._path_in_cache(op, basename)
        _super_open_for_data = super()._open_for_data

        def open_fun(mode):
            return _super_open_for_data(op, basename, mode)

        return cached_open(local_cache_path, mode, open_fun)

    def _open_for_status(self, op: OpSpec, basename: str, mode: str) -> io.IOBase:
        # handling status files differently, so we only cache 'complete' statuses
        return super()._open_for_status(op, basename, mode)

    def _finalize_result(self, op: OpSpec):
        done_path = self._path_in_cache(op, RESULT_INDICATOR)
        with open(done_path, "wt") as f:
            f.write("done")
        super()._finalize_result(op)

    def has_result(self, op: OpSpec) -> bool:
        # Return 'True' quickly, otherwise check cache
        local_path = self._path_in_cache(op, RESULT_INDICATOR)
        log = logger.bind(op=op.uuid, local_path=local_path)
        if op.is_ephemeral:
            # Ephemeral ops are ready only if all dependencies are ready
            return all(dep.has_result(runner=self) for dep in op.get_dependencies())
        if os.path.exists(local_path):
            log.debug("(cached) has_result()", result=True, is_hit=True)
            return True
        if super().has_result(op):
            log.debug("(cached) has_result()", result=True, is_hit=False)
            self._finalize_result(op)
            return True
        else:
            log.debug("(cached) has_result()", result=False)
            return False

    def get_status(self, op: OpSpec) -> OpStatus:
        # Return 'completed' ops quickly, otherwise check cache
        local_path = self._path_in_cache(op, STATUS_JSON_FILE_SUFFIX)
        if os.path.exists(local_path):
            with open(local_path, "rt") as f:
                result = json.load(f)
            # Need to deserialize OpSpec separately
            [result["op"]] = graph_deserialize(result["op"])
            status = OpStatus.model_validate(result)
            if status.state not in {"completed", "ephemeral"}:
                raise RuntimeError(f"Expected completed status, got {status.state}")
            return status
        stat = super().get_status(op)
        if stat.state in {"completed", "ephemeral"}:
            with atomic_write(local_path, "wt") as f:
                f.write(stat.model_dump_json())
        return stat

    def put_status(self, status: OpStatus) -> bool:
        local_path = self._path_in_cache(status.op, STATUS_JSON_FILE_SUFFIX)
        if super().put_status(status):
            if status.state in {"completed", "ephemeral"}:
                with atomic_write(local_path, "wt") as f:
                    f.write(status.model_dump_json())
            return True
        return False


class InMemoryCacheRunner(LocalArrowRunner):
    """A copy-on-write in-memory cache layered on top of another runner.

    Wraps a ``source_runner`` (any :class:`BaseRunner`):

    - **Reads** that aren't satisfied locally fall through to the source.
    - **Writes** live only in this runner's in-memory store and are *never*
      propagated back to the source.

    This is handy for scratch computation on top of an existing (possibly
    remote or expensive) store: run new ops, materialize intermediate results,
    and discard them when this runner is garbage-collected, all without
    mutating the source.

    Implementation note: this extends :class:`LocalArrowRunner` backed by an
    in-memory fsspec ``MemoryFileSystem``. There is no on-disk filesystem --
    ``self.fs`` *is* the in-memory store. Because ``LocalArrowRunner`` routes
    all persistence through ``self.fs``, every inherited write lands in memory
    and nowhere else; we only override the read paths to fall back to the
    source on a local miss.
    """

    def __init__(self, source_runner: BaseRunner):
        """Wrap ``source_runner`` with an in-memory copy-on-write cache."""
        self.source_runner = source_runner
        # fsspec's MemoryFileSystem shares one global store across all
        # instances, so a unique per-instance prefix is required to avoid
        # colliding with other in-memory runners (including a memory:// source).
        super().__init__(store_uri=f"memory://krnel-inmem-cow/{uuid4().hex}")

    def _read(self, op: OpSpec, method: str):
        """Serve a read from the in-memory layer if present, else from source.

        If neither has the result, fall through to ``super()`` so the op is
        materialized locally (its result is then written to memory).
        """
        if not super().has_result(op) and self.source_runner.has_result(op):
            return getattr(self.source_runner, method)(op)
        return getattr(super(), method)(op)

    def to_arrow(self, op: OpSpec):
        return self._read(op, "to_arrow")

    def to_json(self, op: OpSpec) -> dict:
        return self._read(op, "to_json")

    def to_sklearn_estimator(self, op: OpSpec):
        return self._read(op, "to_sklearn_estimator")

    def to_pyobject(self, op: OpSpec):
        return self._read(op, "to_pyobject")

    def has_result(self, op: OpSpec) -> bool:
        if op.is_ephemeral:
            # Ephemeral readiness is dependency-based; recurses via self.
            return super().has_result(op)
        return super().has_result(op) or self.source_runner.has_result(op)

    def get_status(self, op: OpSpec) -> OpStatus:
        # Prefer the in-memory (COW) status. Only consult the source when it
        # actually has the result -- delegating blindly would make the source's
        # get_status() create and persist a "new" status, leaking a write.
        status_path = self._path(op.uuid, STATUS_JSON_FILE_SUFFIX)
        if self.fs.exists(status_path):
            return super().get_status(op)
        if self.source_runner.has_result(op):
            return self.source_runner.get_status(op)
        return super().get_status(op)  # new -> created in memory

    def uuid_to_op(self, uuid: str) -> OpSpec | None:
        op = super().uuid_to_op(uuid)
        if op is not None:
            return op
        op = self.source_runner.uuid_to_op(uuid)
        if op is not None:
            # Rebind to this runner so further work stays in the COW layer
            # rather than routing back to the source.
            op._runner = self
            for dep in op.get_dependencies(recursive=True):
                dep._runner = self
        return op
