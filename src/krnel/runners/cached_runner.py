# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

from typing import Any
from tempfile import gettempdir
import os.path

import fsspec.implementations.cached

from krnel.runners.local_runner import LocalArrowRunner
from krnel.logging import get_logger
logger = get_logger(__name__)

class LocalCachedRunner(LocalArrowRunner):

    def __init__(
        self,
        store_uri: str,
        cache_path: str | None = None,
        expiry_time: int = 24 * 60 * 60 * 7,
        metadata_expiry_time: int = 60,
    ):
        log = logger.bind(store_uri=store_uri, cache_path=cache_path, expiry_time=expiry_time, metadata_expiry_time=metadata_expiry_time)
        fs, _token, paths = fsspec.get_fs_token_paths(store_uri)
        base_path = paths[0]

        if cache_path is not None:
            data_path = os.path.join(cache_path, "data")
            metadata_path = os.path.join(cache_path, "metadata")
        else:
            data_path = os.path.join(
                gettempdir(),
                "krnel_cache",
                "data",
            )
            metadata_path = os.path.join(
                gettempdir(),
                "krnel_cache",
                "metadata",
            )

        cached_fs_data = fsspec.implementations.cached.SimpleCacheFileSystem(
            fs=fs,
            cache_storage=data_path,
            cache_check=metadata_expiry_time,
            expiry_time=expiry_time,
        )
        cached_fs_metadata = fsspec.implementations.cached.SimpleCacheFileSystem(
            fs=fs,
            cache_storage=metadata_path,
            cache_check=metadata_expiry_time,
            expiry_time=metadata_expiry_time,
        )
        log.debug("Initialized cached runner", data_path=data_path, metadata_path=metadata_path, cached_fs=cached_fs_data, cached_fs_metadata=cached_fs_metadata, base_path=base_path)

        super().__init__(store_uri=base_path, filesystem=cached_fs_data, filesystem_for_metadata=cached_fs_metadata)
