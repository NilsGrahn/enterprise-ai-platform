"""Evaluation entrypoint.

CLI: python -m ml_service.evaluate --pipeline credit --version v1 --split test

Loads a saved artifact, scores a held-out split, prints a report, and writes
a model card to docs/architecture/.

Like train.py, this file contains no domain-specific logic.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ml_service.artifacts import load_artifact
from ml_service.config import get_settings
from ml_service.pipelines import get_pipeline


def parse_args():
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument('--pipeline', default=settings.active_pipeline)
    parser.add_argument('--version', default=settings.model_version)
    parser.add_argument('--split', default='test', choices=['train', 'valid', 'test'])
    return parser.parse_args()


def decile_lift_table(y_true, y_score) -> pd.DataFrame:
    """Bucket predictions into deciles and show the actual rate in each.

    A well-ordered model puts far more real positives in the top decile than
    in the bottom one. This is the plainest possible sanity check.
    """
    df = pd.DataFrame({'y': np.asarray(y_true), 'score': np.asarray(y_score)})
    df['decile'] = pd.qcut(df['score'].rank(method='first'), 10, labels=False)
    df['decile'] = 9 - df['decile']  # 0 = highest predicted risk

    base_rate = df['y'].mean()
    table = df.groupby('decile').agg(
        n=('y', 'size'),
        actual_positives=('y', 'sum'),
        actual_rate=('y', 'mean'),
        mean_score=('score', 'mean'),
    ).reset_index()
    table['lift'] = table['actual_rate'] / base_rate
    return table


def build_report(pipeline_name, version, split, metadata, metrics, cm, lift):
    """Return the model card as a markdown string."""
    tn, fp, fn, tp = cm.ravel()
    lines = [
        f"# Model Card — {pipeline_name} {version}",
        "",
        f"- **Evaluated on:** `{split}` split",
        f"- **Generated:** {datetime.now(timezone.utc).isoformat()}",
        f"- **Trained at:** {metadata.get('trained_at', 'unknown')}",
        f"- **Algorithm:** {metadata.get('algorithm', 'unknown')}",
        f"- **Training rows:** {metadata.get('training_rows', 'unknown'):,}",
        f"- **Features:** {len(metadata.get('feature_names', []))}",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| AUC | {metrics['auc']:.4f} |",
        f"| KS | {metrics['ks']:.4f} |",
        f"| Precision | {metrics['precision']:.4f} |",
        f"| Recall | {metrics['recall']:.4f} |",
        f"| F1 | {metrics['f1']:.4f} |",
        f"| Threshold | {metrics['threshold']:.2f} |",
        "",
        "## Confusion matrix",
        "",
        "| | Predicted 0 | Predicted 1 |",
        "|---|---|---|",
        f"| **Actual 0** | {tn:,} | {fp:,} |",
        f"| **Actual 1** | {fn:,} | {tp:,} |",
        "",
        "## Decile lift",
        "",
        "Decile 0 is the highest predicted risk.",
        "",
        "| Decile | n | Actual positives | Actual rate | Mean score | Lift |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in lift.iterrows():
        lines.append(
            f"| {int(r['decile'])} | {int(r['n']):,} | {int(r['actual_positives']):,} "
            f"| {r['actual_rate']:.4f} | {r['mean_score']:.4f} | {r['lift']:.2f}x |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    settings = get_settings()

    model, metadata = load_artifact(args.pipeline, args.version)

    pipeline = get_pipeline(
        args.pipeline,
        preprocessing=metadata['preprocessing'],
    )
    pipeline.model = model

    df = pipeline.load_data(args.split)
    print(f"loaded {len(df)} rows from the '{args.split}' split")

    pipeline.validate_schema(df)
    cleaned = pipeline.clean(df)
    y_true = cleaned[pipeline.target_column].astype(int)

    X, _ = pipeline.feature_engineering(cleaned, fit=False)
    y_score = model.predict_proba(X)[:, 1]

    threshold = settings.decision_threshold
    y_pred = (y_score >= threshold).astype(int)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    metrics = {
        'auc': float(roc_auc_score(y_true, y_score)),
        'ks': float(np.max(tpr - fpr)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'threshold': float(threshold),
    }

    cm = confusion_matrix(y_true, y_pred)
    lift = decile_lift_table(y_true, y_score)

    print()
    print(f"--- {args.pipeline} {args.version} on '{args.split}' ---")
    for k, v in metrics.items():
        print(f"  {k:<12} {v:>10.4f}")
    print()
    print("confusion matrix (rows = actual, cols = predicted):")
    print(cm)
    print()
    print("decile lift:")
    print(lift.to_string(index=False))

    training_metrics = metadata.get('metrics', {})
    if 'auc' in training_metrics:
        gap = training_metrics['auc'] - metrics['auc']
        print()
        print(f"validation AUC {training_metrics['auc']:.4f} vs "
              f"{args.split} AUC {metrics['auc']:.4f} (gap {gap:+.4f})")
        if abs(gap) > 0.02:
            print("  WARNING: gap above 0.02 suggests overfitting to the "
                  "validation set through early stopping.")
        else:
            print("  gap is within the expected range.")

    report = build_report(args.pipeline, args.version, args.split,
                          metadata, metrics, cm, lift)
    out_dir = Path('docs/architecture')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"model_card_{args.pipeline}_{args.version}.md"
    out_path.write_text(report)
    print(f"\nwrote {out_path}")


if __name__ == '__main__':
    main()