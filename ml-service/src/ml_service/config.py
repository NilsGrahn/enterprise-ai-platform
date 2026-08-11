from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class MLSettings(BaseSettings):
    """Typed configuration for the ML service, loaded from .env.

    Every attribute below is read from an environment variable of the same
    name, upper-cased. ACTIVE_PIPELINE -> active_pipeline, and so on.
    """

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
        protected_namespaces=(),
    )

    active_pipeline: str = 'credit'
    artifact_dir: Path = Path('./artifacts')
    model_version: str = 'v1'
    random_seed: int = 42
    decision_threshold: float = 0.5


@lru_cache(maxsize=1)
def get_settings() -> MLSettings:
    """Return the settings object, constructed once and reused."""
    return MLSettings()