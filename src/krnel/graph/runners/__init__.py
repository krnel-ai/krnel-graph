# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

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


def Runner(*, type: str | None = None, **kwargs) -> LocalArrowRunner:  # noqa: N802
    """Create a runner instance with configuration from environment, file, or parameters.

    Args:
        type: Runner type (e.g., 'LocalCachedRunner'). If None, uses configuration.
        **kwargs: Additional parameters to pass to the runner constructor.

    Returns:
        Configured runner instance.

    Configuration priority:
        1. Explicit parameters (type and kwargs)
        2. Environment variables (KRNEL_RUNNER_TYPE, KRNEL_RUNNER_STORE_URI)
        3. JSON config file (~/.config/krnel/graph_runner_cfg.json)
        4. Default values

    Raises:
        ValueError: If no store_uri is provided for LocalCachedRunner.
    """
    from krnel.graph.op_spec import find_subclass_of

    # If no explicit type/kwargs provided, use configuration
    if type is None:
        config = KrnelGraphConfig()
        type = config.runner_type

    runner_class = find_subclass_of(BaseRunner, type)
    if runner_class is None:
        raise ValueError(f"Unknown runner type: {type!r}")
    return runner_class()
