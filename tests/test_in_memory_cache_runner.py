# Copyright (c) 2025-2026 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

# ruff: noqa: S101

import uuid

import pytest

from krnel.graph.dataset_ops import (
    AssignTrainTestSplitOp,
    LoadInlineJsonDatasetOp,
)
from krnel.graph.runners.cached_runner import InMemoryCacheRunner
from krnel.graph.runners.local_runner import LocalArrowRunner


@pytest.fixture
def sample_dataset():
    data = {
        "id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "value": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
    }
    return LoadInlineJsonDatasetOp(data=data)


@pytest.fixture
def source():
    """A backing runner with its own (isolated) in-memory store."""
    return LocalArrowRunner(store_uri=f"memory://src-{uuid.uuid4().hex}")


def test_read_proxied_from_source(source, sample_dataset):
    """A persisted result in the source is served through the cache."""
    # AssignTrainTestSplitOp is non-ephemeral, so the source stores a result.
    op = AssignTrainTestSplitOp(dataset=sample_dataset, test_size=0.5, random_state=3)
    expected = source.to_arrow(op)
    assert source.has_result(op)

    cache = InMemoryCacheRunner(source)

    assert cache.has_result(op)
    assert cache.to_arrow(op).equals(expected)
    assert cache.get_status(op).state == "completed"


def test_writes_do_not_leak_to_source(source, sample_dataset):
    """Materializing a new (persisted) op via the cache must not touch source."""
    cache = InMemoryCacheRunner(source)

    # A non-ephemeral op the source has never seen.
    new_op = AssignTrainTestSplitOp(
        dataset=sample_dataset, test_size=0.5, random_state=1
    )
    assert not source.has_result(new_op)

    cache.to_arrow(new_op)  # materialize via the cache

    # Result lives in the cache only.
    assert cache.has_result(new_op)
    assert not source.has_result(new_op)
    # And no status leaked into the source.
    assert source.uuid_to_op(new_op.uuid) is None


def test_status_read_does_not_leak_to_source(source, sample_dataset):
    """get_status() for an unknown op creates status in memory, not in source."""
    cache = InMemoryCacheRunner(source)
    new_op = AssignTrainTestSplitOp(
        dataset=sample_dataset, test_size=0.25, random_state=7
    )

    status = cache.get_status(new_op)
    assert status.state == "new"
    assert source.uuid_to_op(new_op.uuid) is None
