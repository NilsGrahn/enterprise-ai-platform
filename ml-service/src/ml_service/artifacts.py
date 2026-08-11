import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

from ml_service.config import get_settings


class ArtifactNotFoundError(Exception):
    """Raised when a requested model artifact does not exist on disk."""


def artifact_path(pipeline_name: str, version: str) -> Path:
    """Return the directory where this pipeline/version's artifact lives.

    Layout: artifacts/{pipeline_name}/{version}/
    """
    settings = get_settings()
    return settings.artifact_dir / pipeline_name / version


def save_artifact(pipeline_name, version, model, metadata, feature_names):
    """Write model.pkl and metadata.json for one trained model.

    metadata is expected to contain: algorithm, hyperparameters,
    training_rows, metrics, preprocessing.
    """
    path = artifact_path(pipeline_name, version)
    path.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, path / 'model.pkl')

    full_metadata = {
        'pipeline_name': pipeline_name,
        'model_version': version,
        'trained_at': datetime.now(timezone.utc).isoformat(),
        'feature_names': list(feature_names),
        **metadata,
    }

    with open(path / 'metadata.json', 'w') as f:
        json.dump(full_metadata, f, indent=2, default=str)

    return path


def load_artifact(pipeline_name, version):
    """Load a saved model and its metadata.

    Returns (model, metadata). Raises ArtifactNotFoundError if either file
    is missing.
    """
    path = artifact_path(pipeline_name, version)
    model_file = path / 'model.pkl'
    metadata_file = path / 'metadata.json'

    if not model_file.exists() or not metadata_file.exists():
        raise ArtifactNotFoundError(
            f"No artifact for pipeline '{pipeline_name}' version '{version}'. "
            f"Looked in {path}. "
            f"Train it first with: python -m ml_service.train "
            f"--pipeline {pipeline_name} --version {version}"
        )

    model = joblib.load(model_file)
    with open(metadata_file) as f:
        metadata = json.load(f)

    return model, metadata