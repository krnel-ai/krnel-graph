# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

import contextlib
import io
import json
import os.path

import fsspec
import fsspec.implementations.cached
from fsspec.utils import atomic_write

from krnel.graph import config
from krnel.graph.op_spec import OpSpec, graph_deserialize
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
            return True  # Ephemeral ops are always "available"
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
            if status.state != "completed":
                raise RuntimeError(f"Expected completed status, got {status.state}")
            return status
        stat = super().get_status(op)
        if stat.state == "completed":
            with atomic_write(local_path, "wt") as f:
                f.write(stat.model_dump_json())
        return stat

    def put_status(self, status: OpStatus) -> bool:
        if status.op.is_ephemeral:
            # Ephemeral ops do not have a status file, they are always 'ephemeral'
            return True
        local_path = self._path_in_cache(status.op, STATUS_JSON_FILE_SUFFIX)
        if super().put_status(status):
            if status.state == "completed":
                with atomic_write(local_path, "wt") as f:
                    f.write(status.model_dump_json())
            return True
        return False
