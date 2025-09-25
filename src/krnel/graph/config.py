# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

import tempfile
from pathlib import Path

from platformdirs import user_config_dir
from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    JsonConfigSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class KrnelGraphConfig(BaseSettings):
    """Configuration for graph runners.

    Configuration sources in priority order:
    1. Environment variables (KRNEL_RUNNER_TYPE, KRNEL_RUNNER_STORE_URI)
    2. JSON config file at ~/.config/krnel/graph_runner_cfg.json
    3. Default values
    """

    model_config = SettingsConfigDict(
        env_prefix="KRNEL_",
        case_sensitive=False,
        extra="ignore",
        json_file=Path(user_config_dir("krnel")) / "krnel_graph_config.json",
    )

    runner_type: str = Field(
        default="LocalRunner",
        description="Type of runner to use (e.g., 'LocalCachedRunner', 'LocalArrowRunner')",
    )

    store_uri: str = Field(
        default=str(Path(tempfile.gettempdir()) / "krnel"),
        description="Where all graph data is cached. (Large-scale shared storage, e.g., '/tmp/', 'gs://bucket/path-to-storage').",
    )

    cache_path: Path = Field(
        default=Path(tempfile.gettempdir()) / "krnel_cache",
        description="Local path to use for caching. (runner_type=LocalCacheRunner only)",
    )

    def save(self):
        """Save current configuration to JSON file."""
        config_path = self.model_config["json_file"]
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            val = self.model_dump_json(exclude_defaults=True, indent=4)
            f.write(val + "\n")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            file_secret_settings,
            JsonConfigSettingsSource(settings_cls),
        )
