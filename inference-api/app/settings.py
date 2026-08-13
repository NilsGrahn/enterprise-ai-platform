from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# app/settings.py -> app/ -> inference-api/ -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]


class APISettings(BaseSettings):
    """Typed configuration for the inference API.

    The env_file path is resolved from this file's own location, so the
    settings load correctly whether the app is started from the repo root,
    from inference-api/, or from inside a container.
    """

    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / '.env'),
        env_file_encoding='utf-8',
        extra='ignore',
        protected_namespaces=(),
    )

    active_pipeline: str = 'credit'
    model_version: str = 'v1'
    artifact_dir: Path = REPO_ROOT / 'artifacts'
    decision_threshold: float = 0.5
    llm_enabled: bool = True

    app_name: str = 'inference-api'
    app_version: str = '0.1.0'

    max_batch_size: int = 500
    metrics_cache_seconds: int = 10


@lru_cache(maxsize=1)
def get_settings() -> APISettings:
    return APISettings()