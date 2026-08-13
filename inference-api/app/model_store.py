import time

from sqlalchemy import text

from data_platform.db import get_engine
from explain_service.explainer import ShapTreeExplainer
from llm_service.client import LLMClient, NullLLMClient
from ml_service.artifacts import load_artifact, ArtifactNotFoundError
from ml_service.pipelines import get_pipeline


class ModelStore:
    """Holds everything loaded once at startup and reused for every request."""

    def __init__(self):
        self.pipeline = None
        self.metadata = None
        self.explainer = None
        self.llm_client = None
        self.model_key = None
        self.loaded_at = None
        self.load_error = None
        self.started_at = time.monotonic()

    def load(self, settings):
        """Load the active model. Records the error rather than raising, so the
        service can start and report itself unhealthy instead of crash-looping."""
        try:
            print(f"[model_store] loading {settings.active_pipeline} "
                  f"{settings.model_version} from {settings.artifact_dir}")

            model, metadata = load_artifact(
                settings.active_pipeline, settings.model_version
            )

            pipeline = get_pipeline(
                settings.active_pipeline,
                preprocessing=metadata['preprocessing'],
            )
            pipeline.model = model

            self.pipeline = pipeline
            self.metadata = metadata
            self.explainer = ShapTreeExplainer(pipeline, metadata)
            self.llm_client = LLMClient() if settings.llm_enabled else NullLLMClient()
            self.model_key = self._resolve_model_key(
                settings.active_pipeline, settings.model_version
            )
            self.loaded_at = time.time()
            self.load_error = None

            print(f"[model_store] ready — {len(metadata['feature_names'])} features, "
                  f"model_key={self.model_key}, "
                  f"llm={'enabled' if settings.llm_enabled else 'disabled'}")

        except (ArtifactNotFoundError, Exception) as exc:
            self.load_error = str(exc)
            print(f"[model_store] LOAD FAILED: {exc}")

    @staticmethod
    def _resolve_model_key(pipeline_name, model_version):
        """Look up the dim_model row for this version. None if unavailable."""
        try:
            engine = get_engine()
            with engine.begin() as conn:
                row = conn.execute(text("""
                    SELECT model_key FROM gold.dim_model
                    WHERE pipeline_name = :p AND model_version = :v
                """), {'p': pipeline_name, 'v': model_version}).fetchone()
            return int(row[0]) if row else None
        except Exception as exc:
            print(f"[model_store] could not resolve model_key: {exc}")
            return None

    def is_ready(self) -> bool:
        return self.pipeline is not None and self.pipeline.model is not None

    def uptime_seconds(self) -> float:
        return time.monotonic() - self.started_at


MODEL_STORE = ModelStore()