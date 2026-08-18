import json

import numpy as np
import pandas as pd

DEFAULT_BINS = 10
EPSILON = 1e-6          # replaces zero shares, so ln() and division stay finite


def population_stability_index(reference, current, bin_edges=None,
                               bins=DEFAULT_BINS) -> float:
    """PSI between a reference and a current distribution for one feature.

    bin_edges: boundaries computed from the REFERENCE data at training time.
               If omitted they are derived from `reference` here, which is only
               correct when `reference` really is the frozen training sample.
    """
    reference = pd.Series(reference).dropna().astype(float)
    current = pd.Series(current).dropna().astype(float)

    if len(reference) == 0 or len(current) == 0:
        return 0.0

    if bin_edges is None:
        bin_edges = quantile_bin_edges(reference, bins=bins)

    edges = np.asarray(bin_edges, dtype=float)

    reference_counts, _ = np.histogram(reference, bins=edges)
    current_counts, _ = np.histogram(current, bins=edges)

    reference_share = reference_counts / max(reference_counts.sum(), 1)
    current_share = current_counts / max(current_counts.sum(), 1)

    reference_share = np.where(reference_share == 0, EPSILON, reference_share)
    current_share = np.where(current_share == 0, EPSILON, current_share)

    psi = np.sum((current_share - reference_share)
                 * np.log(current_share / reference_share))

    return float(psi)


def quantile_bin_edges(values, bins=DEFAULT_BINS) -> list:
    """Bin boundaries at the quantiles of `values`, with open outer edges.

    Duplicate interior edges are removed — a feature that is mostly one value
    (many of the engineered flags are) would otherwise produce zero-width bins
    and a divide-by-zero.
    """
    series = pd.Series(values).dropna().astype(float)
    if len(series) == 0:
        return [-np.inf, np.inf]

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(series.quantile(quantiles).to_numpy())

    if len(edges) < 2:
        # Constant feature: one bin covering everything.
        return [-np.inf, np.inf]

    edges = edges.astype(float)
    edges[0] = -np.inf         # catch values below the training minimum
    edges[-1] = np.inf         # catch values above the training maximum
    return edges.tolist()


def classify(psi, warn=0.10, alert=0.25) -> str:
    """Map a PSI value to OK / WARN / ALERT."""
    if psi >= alert:
        return 'ALERT'
    if psi >= warn:
        return 'WARN'
    return 'OK'


def build_reference_profile(pipeline, metadata, split='train',
                            bins=DEFAULT_BINS) -> dict:
    """Compute the frozen reference distribution for every engineered feature.

    Runs the training split through the pipeline's own clean() and
    feature_engineering(fit=False), so the reference describes exactly what
    the model consumed during training.
    """
    raw = pipeline.load_data(split)
    cleaned = pipeline.clean(raw)
    X, _ = pipeline.feature_engineering(cleaned, fit=False)

    features = {}
    for column in X.columns:
        values = X[column]
        edges = quantile_bin_edges(values, bins=bins)
        counts, _ = np.histogram(
            pd.Series(values).dropna().astype(float),
            bins=np.asarray(edges, dtype=float),
        )
        total = max(int(counts.sum()), 1)
        features[column] = {
            'bin_edges': edges,
            'reference_share': (counts / total).tolist(),
            'n_reference': int(total),
            'mean': float(values.mean()),
            'std': float(values.std()),
        }

    return {
        'pipeline_name': metadata['pipeline_name'],
        'model_version': metadata['model_version'],
        'split': split,
        'bins': bins,
        'feature_names': list(X.columns),
        'features': features,
    }


def psi_from_profile(feature_profile, current_values) -> float:
    """PSI for one feature, using its stored reference shares and bin edges."""
    current = pd.Series(current_values).dropna().astype(float)
    if len(current) == 0:
        return 0.0

    edges = np.asarray(feature_profile['bin_edges'], dtype=float)
    reference_share = np.asarray(feature_profile['reference_share'], dtype=float)

    current_counts, _ = np.histogram(current, bins=edges)
    current_share = current_counts / max(current_counts.sum(), 1)

    reference_share = np.where(reference_share == 0, EPSILON, reference_share)
    current_share = np.where(current_share == 0, EPSILON, current_share)

    return float(np.sum((current_share - reference_share)
                        * np.log(current_share / reference_share)))