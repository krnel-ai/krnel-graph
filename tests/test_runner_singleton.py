# Copyright (c) 2025-2026 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

"""Test suite for Runner singleton behavior."""

import gc
import tempfile
import warnings
import weakref
from pathlib import Path

import pytest

from krnel.graph.runners import (
    Runner,
    LocalArrowRunner,
    LocalCachedRunner,
    _RUNNER_REGISTRY,
)


@pytest.fixture(autouse=True)
def reset_runner_registry():
    """Clear the runner registry before each test to ensure test isolation."""
    global _RUNNER_REGISTRY
    # Import the module to reset the warning flag
    import krnel.graph.runners as runners_module

    # Clear registry
    _RUNNER_REGISTRY.clear()

    # Reset warning flag
    runners_module._MULTIPLE_RUNNERS_WARNING_EMITTED = False

    yield

    # Cleanup after test
    _RUNNER_REGISTRY.clear()
    runners_module._MULTIPLE_RUNNERS_WARNING_EMITTED = False


def test_runner_singleton_same_params():
    """Calling Runner() twice with same params returns same instance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runner1 = Runner(store_uri=tmpdir)
        runner2 = Runner(store_uri=tmpdir)

        # Should be the exact same instance
        assert runner1 is runner2


def test_runner_singleton_different_params():
    """Creating runners with different params emits warning."""
    with tempfile.TemporaryDirectory() as tmpdir1:
        with tempfile.TemporaryDirectory() as tmpdir2:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                runner1 = Runner(store_uri=tmpdir1)
                runner2 = Runner(store_uri=tmpdir2)

                # Should be different instances
                assert runner1 is not runner2

                # Should emit warning about multiple runners
                assert len(w) == 1
                assert "Multiple distinct runner contexts" in str(w[0].message)
                assert "runner.to_json(op)" in str(w[0].message)


def test_runner_singleton_exact_match():
    """Create runners with identical arguments returns same instance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create runners with exactly the same arguments
        runner1 = Runner(store_uri=tmpdir)
        runner2 = Runner(store_uri=tmpdir)

        # Should be the same instance (exact match)
        assert runner1 is runner2

        # Different string (even if semantically equivalent) creates different runner
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            runner3 = Runner(store_uri=tmpdir + "/")  # trailing slash is different string

            # Should be different instance (different string argument)
            assert runner1 is not runner3

            # Should emit warning
            assert len(w) == 1


def test_runner_singleton_different_types():
    """Create LocalArrowRunner and LocalCachedRunner."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with tempfile.TemporaryDirectory() as cache_dir:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                runner1 = Runner(type="LocalArrowRunner", store_uri=tmpdir)
                runner2 = Runner(type="LocalCachedRunner", store_uri=tmpdir, cache_path=cache_dir)

                # Should be different instances
                assert runner1 is not runner2
                assert isinstance(runner1, LocalArrowRunner)
                assert isinstance(runner2, LocalCachedRunner)

                # Should emit warning
                assert len(w) == 1
                assert "Multiple distinct runner contexts" in str(w[0].message)


def test_runner_singleton_filesystem_comparison():
    """Create runners with same store_uri but different filesystems."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create two runners with the same store_uri (will use same filesystem)
        runner1 = Runner(store_uri=tmpdir)
        runner2 = Runner(store_uri=tmpdir)

        # Should be the same instance
        assert runner1 is runner2

        # Create a runner with memory filesystem (suppress warning)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            runner3 = Runner(store_uri="memory://test")

        # Should be different from the local filesystem runners
        assert runner1 is not runner3


def test_runner_singleton_warning_once():
    """Create 3 different runners in sequence."""
    with tempfile.TemporaryDirectory() as tmpdir1:
        with tempfile.TemporaryDirectory() as tmpdir2:
            with tempfile.TemporaryDirectory() as tmpdir3:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")

                    runner1 = Runner(store_uri=tmpdir1)
                    runner2 = Runner(store_uri=tmpdir2)
                    runner3 = Runner(store_uri=tmpdir3)

                    # Should have 3 different instances
                    assert runner1 is not runner2
                    assert runner2 is not runner3
                    assert runner1 is not runner3

                    # Should emit warning only once (not 3 times)
                    assert len(w) == 1
                    assert "Multiple distinct runner contexts" in str(w[0].message)


def test_runner_singleton_with_memory_filesystem():
    """Test singleton behavior with memory:// URIs."""
    runner1 = Runner(store_uri="memory://test1")
    runner2 = Runner(store_uri="memory://test1")

    # Same URI should return same instance
    assert runner1 is runner2

    # Different URI should create different instance
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        runner3 = Runner(store_uri="memory://test2")

        assert runner1 is not runner3
        assert len(w) == 1


def test_runner_singleton_with_complex_kwargs():
    """Test singleton behavior with complex nested kwargs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create runner with same kwargs
        runner1 = Runner(type="LocalCachedRunner", store_uri=tmpdir, cache_path="/tmp/cache1")
        runner2 = Runner(type="LocalCachedRunner", store_uri=tmpdir, cache_path="/tmp/cache1")

        # Same kwargs should return same instance
        assert runner1 is runner2

        # Different kwargs should create different instance
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            runner3 = Runner(type="LocalCachedRunner", store_uri=tmpdir, cache_path="/tmp/cache2")

            assert runner1 is not runner3
            assert len(w) == 1


def test_runner_garbage_collection():
    """Test that runners can be garbage collected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a runner and get weak reference
        runner = Runner(store_uri=tmpdir)
        weak_ref = weakref.ref(runner)

        # Verify runner exists
        assert weak_ref() is not None
        assert len(_RUNNER_REGISTRY) == 1

        # Delete strong references
        del runner

        # Force garbage collection
        gc.collect()

        # Verify runner was collected
        assert weak_ref() is None

        # Registry should be empty (weak references are automatically removed)
        assert len(_RUNNER_REGISTRY) == 0

        # Create new runner with same params - should be a new instance
        runner2 = Runner(store_uri=tmpdir)
        assert weak_ref() is None  # Old reference still None
        assert len(_RUNNER_REGISTRY) == 1


def test_runner_kwargs_passthrough():
    """Test that kwargs are actually passed to runner constructor."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create runner with explicit store_uri
        runner = Runner(store_uri=tmpdir)

        # Verify the store_uri was passed through
        assert isinstance(runner, LocalArrowRunner)
        # The _store_uri should be set to tmpdir
        assert runner._store_uri == tmpdir

        # Test with LocalCachedRunner (suppress warning from creating second runner)
        with tempfile.TemporaryDirectory() as cache_dir:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cached_runner = Runner(
                    type="LocalCachedRunner",
                    store_uri=tmpdir,
                    cache_path=cache_dir
                )

            # Verify both parameters were passed through
            assert isinstance(cached_runner, LocalCachedRunner)
            assert cached_runner._store_uri == tmpdir
            assert Path(cached_runner.cache_path).resolve() == Path(cache_dir).resolve()
