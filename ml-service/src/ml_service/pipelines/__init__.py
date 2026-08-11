"""Pipeline registry.

To add a new domain:
  1. Drop {domain}_pipeline.py into this folder
  2. Import it below and add one line to PIPELINE_REGISTRY
  3. Set ACTIVE_PIPELINE={domain} in .env

Nothing else in the repository changes.
"""

from ml_service.base_pipeline import BasePipeline
from ml_service.pipelines.credit_pipeline import CreditPipeline


class UnknownPipelineError(Exception):
    """Raised when a pipeline name is not in the registry."""


PIPELINE_REGISTRY: dict[str, type[BasePipeline]] = {
    'credit': CreditPipeline,
}


def get_pipeline(name: str, **kwargs) -> BasePipeline:
    """Construct the pipeline registered under `name`."""
    if name not in PIPELINE_REGISTRY:
        raise UnknownPipelineError(
            f"Unknown pipeline '{name}'. Available: {sorted(PIPELINE_REGISTRY)}"
        )
    return PIPELINE_REGISTRY[name](**kwargs)