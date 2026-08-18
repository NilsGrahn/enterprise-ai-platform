"""Build the frozen reference profile for the active model.

CLI: python -m monitoring.build_reference --pipeline credit --version v1

Run once, immediately after training. Drift is meaningless without a frozen
reference — recomputing bin edges from current data would make PSI always zero.
"""

import argparse
import json

from ml_service.artifacts import artifact_path, load_artifact
from ml_service.config import get_settings
from ml_service.pipelines import get_pipeline
from monitoring.drift import build_reference_profile


def parse_args():
    settings = get_settings()
    parser = argparse.ArgumentParser(description='Build a drift reference profile.')
    parser.add_argument('--pipeline', default=settings.active_pipeline)
    parser.add_argument('--version', default=settings.model_version)
    parser.add_argument('--split', default='train', choices=['train', 'valid', 'test'])
    parser.add_argument('--bins', type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()

    model, metadata = load_artifact(args.pipeline, args.version)
    pipeline = get_pipeline(args.pipeline, preprocessing=metadata['preprocessing'])
    pipeline.model = model

    print(f"building reference profile from the '{args.split}' split…")
    profile = build_reference_profile(pipeline, metadata,
                                     split=args.split, bins=args.bins)

    out_path = artifact_path(args.pipeline, args.version) / 'reference_profile.json'
    with open(out_path, 'w') as f:
        json.dump(profile, f, indent=2, default=str)

    print(f"wrote {out_path}")
    print(f"features profiled: {len(profile['feature_names'])}")
    print(f"reference rows:    "
          f"{profile['features'][profile['feature_names'][0]]['n_reference']:,}")


if __name__ == '__main__':
    main()