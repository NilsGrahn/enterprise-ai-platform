"""Training entrypoint.

CLI: python -m ml_service.train --pipeline credit --version v1 --activate

Note: this file contains NO domain-specific logic. It works unchanged for
any pipeline in the registry. That is the acceptance criterion for the
platform abstraction.
"""

import argparse

from ml_service.config import get_settings
from ml_service.pipelines import get_pipeline
from ml_service.registry import activate_model


def parse_args():
    settings = get_settings()

    parser = argparse.ArgumentParser(description="Train a model pipeline.")
    parser.add_argument(
        '--pipeline',
        default=settings.active_pipeline,
        help="Pipeline name from the registry (default: ACTIVE_PIPELINE from .env)",
    )
    parser.add_argument(
        '--version',
        default=settings.model_version,
        help="Model version label (default: MODEL_VERSION from .env)",
    )
    parser.add_argument(
        '--activate',
        action='store_true',
        help="Make this model the active one for its pipeline after training",
    )
    return parser.parse_args()


def print_summary(pipeline_name, version, result):
    m = result.metrics
    print()
    print("=" * 52)
    print(f"  Training complete: {pipeline_name} {version}")
    print("=" * 52)
    print(f"  {'training rows':<20} {result.training_rows:>12,}")
    print(f"  {'features':<20} {len(result.feature_names):>12}")
    print(f"  {'best iteration':<20} {m.get('best_iteration', 0):>12}")
    print("-" * 52)
    print(f"  {'AUC':<20} {m['auc']:>12.4f}")
    print(f"  {'KS':<20} {m['ks']:>12.4f}")
    print(f"  {'precision':<20} {m['precision']:>12.4f}")
    print(f"  {'recall':<20} {m['recall']:>12.4f}")
    print(f"  {'threshold':<20} {m['threshold']:>12.2f}")
    print("=" * 52)


def main():
    args = parse_args()
    settings = get_settings()

    if args.version != settings.model_version:
        settings.model_version = args.version

    print(f"training pipeline '{args.pipeline}' as version '{args.version}'")

    pipeline = get_pipeline(args.pipeline)
    result = pipeline.run_training()

    print_summary(args.pipeline, args.version, result)

    if args.activate:
        model_key = activate_model(args.pipeline, args.version)
        print(f"activated model_key={model_key} for pipeline '{args.pipeline}'")
    else:
        print("not activated (pass --activate to make this the live model)")


if __name__ == '__main__':
    main()