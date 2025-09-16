# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

from krnel.graph.runners.local_runner import LocalArrowRunner
from krnel.graph.runners.base_runner import BaseRunner
from krnel.graph.runners.cached_runner import LocalCachedRunner
from krnel.graph.runners.model_registry import ModelProvider, register_model_provider, get_model_provider
from krnel.graph.runners.op_status import OpStatus
from platformdirs import user_config_dir
from pathlib import Path

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

RUNNER_CONFIG_ENV_VAR = "KRNEL_GRAPH_RUNNER_CONFIG"
RUNNER_CONFIG_JSON_PATH = Path(user_config_dir("krnel")) / "graph_runner_cfg.json"

def Runner(type: str | None = None, **kwargs) -> BaseRunner:
    import json
    import os
    from krnel.graph.op_spec import find_subclass_of

    if type == None and kwargs == {}:
        # Try to load from environment variable
        env_json = os.getenv(RUNNER_CONFIG_ENV_VAR, None)
        if env_json is not None:
            config = json.loads(env_json)
            type = config.get("type", None)
            if "type" in config:
                del config["type"]
            kwargs = config
        else:
            # Try to load from config file
            config_file_name = RUNNER_CONFIG_JSON_PATH
            if config_file_name.exists():
                with open(config_file_name, "r") as f:
                    config = json.load(f)
                    type = config.get("type", None)
                    del config["type"]
                    kwargs = config
            else:
                raise ValueError("No graph runner configuration provided. Please specify type and parameters, or set KRNEL_GRAPH_CONFIG environment variable, or create a config file at " + str(config_file_name) + " with content {\"store_uri\": \"gs://bucket/path-to-storage\"}")

    if type == None:
        type = "LocalArrowRunner"

    runner_class = find_subclass_of(BaseRunner, type)
    return runner_class(**kwargs)

