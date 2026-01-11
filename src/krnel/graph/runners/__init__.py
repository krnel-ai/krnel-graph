# Copyright (c) 2025-2026 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

import warnings
import weakref

from krnel.graph.config import KrnelGraphConfig
from krnel.graph.runners.base_runner import BaseRunner
from krnel.graph.runners.cached_runner import LocalCachedRunner
from krnel.graph.runners.local_runner import LocalArrowRunner
from krnel.graph.runners.model_registry import (
    ModelProvider,
    get_model_provider,
    register_model_provider,
)
from krnel.graph.runners.op_status import OpStatus

__all__ = [
    "LocalArrowRunner",
    "LocalCachedRunner",
    "BaseRunner",
    "OpStatus",
    "ModelProvider",
    "register_model_provider",
    "get_model_provider",
    "Runner",
]

# Registry of runner instances (weak references for garbage collection)
_RUNNER_REGISTRY: weakref.WeakValueDictionary = weakref.WeakValueDictionary()

# Track if warning has been emitted (only warn once per process)
_MULTIPLE_RUNNERS_WARNING_EMITTED = False


def _make_kwargs_hashable(kwargs: dict) -> frozenset:
    """Convert kwargs dict to a hashable frozenset for use as registry key.

    Handles nested dicts and lists by converting them to tuples.
    """
    def make_hashable(value):
        if isinstance(value, dict):
            return frozenset((k, make_hashable(v)) for k, v in sorted(value.items()))
        elif isinstance(value, list):
            return tuple(make_hashable(v) for v in value)
        elif isinstance(value, set):
            return frozenset(make_hashable(v) for v in value)
        else:
            return value

    return frozenset((k, make_hashable(v)) for k, v in sorted(kwargs.items()))


def Runner(*, type: str | None = None, **kwargs) -> BaseRunner:  # noqa: N802
    """Create a runner instance with configuration from environment, file, or parameters.

    Runners are cached based on their type and constructor parameters, so multiple calls
    with the same parameters will return the same instance. This ensures that all operations derive from the same runner instance.

    Args:
        type: Runner type (e.g., 'LocalCachedRunner'). If None, uses configuration.
        **kwargs: Additional parameters to pass to the runner constructor.

    Returns:
        Runner instance (may be cached from previous call with same params).

    Configuration priority:
        1. Explicit parameters (type and kwargs)
        2. Environment variables (KRNEL_RUNNER_TYPE, KRNEL_RUNNER_STORE_URI)
        3. JSON config file (~/.config/krnel/graph_runner_cfg.json)
        4. Default values

    Raises:
        ValueError: If unknown runner type specified.
    """
    global _MULTIPLE_RUNNERS_WARNING_EMITTED
    from krnel.graph.op_spec import find_subclass_of

    # Determine runner type
    if type is None:
        config = KrnelGraphConfig()
        type = config.runner_type

    runner_class = find_subclass_of(BaseRunner, type)
    if runner_class is None:
        raise ValueError(f"Unknown runner type: {type!r}")

    registry_key = (runner_class, type, _make_kwargs_hashable(kwargs))

    if registry_key in _RUNNER_REGISTRY:
        return _RUNNER_REGISTRY[registry_key]

    runner = runner_class(**kwargs)
    _RUNNER_REGISTRY[registry_key] = runner
    if len(_RUNNER_REGISTRY) > 1 and not _MULTIPLE_RUNNERS_WARNING_EMITTED:
        _MULTIPLE_RUNNERS_WARNING_EMITTED = True
        warnings.warn(
            "Multiple distinct runner contexts detected. Operations from different runners "
            "cannot be mixed in the same graph. You'll need to use runner.to_json(op) "
            "syntax instead of op.to_json() when working with operations from multiple runners.",
            UserWarning,
            stacklevel=2
        )

    return runner
