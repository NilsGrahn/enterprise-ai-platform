from abc import ABC, abstractmethod

import numpy as np
import shap

from explain_service.schema import ExplanationResult, FeatureContribution


class MissingDisplayNameError(Exception):
    """Raised when an engineered feature has no human-readable label."""


DISPLAY_NAMES = {
    # raw / cleaned inputs
    'income_missing':           'Income not provided',
    'monthly_income':           'Monthly income',
    'revolving_utilisation':    'Credit utilisation ratio',
    'debt_ratio':               'Debt-to-income ratio',
    'age':                      'Age',
    'number_of_dependents':     'Number of dependents',
    'open_credit_lines':        'Open credit accounts',
    'real_estate_loans':        'Real estate loans or lines',
    'times_30_59_days_late':    'Payments 30-59 days late',
    'times_60_89_days_late':    'Payments 60-89 days late',
    'times_90_days_late':       'Payments 90+ days late',
    'total_delinquency_events': 'Total late payment events',
    # derived features
    'monthly_debt_payment':     'Estimated monthly debt payment',
    'income_per_dependent':     'Income per household member',
    'delinquency_ratio':        'Late payments per credit account',
    'has_any_delinquency':      'Any history of late payment',
    'utilisation_bucket':       'Credit utilisation band',
}


class BaseExplainer(ABC):
    """Contract for any explanation method."""

    @abstractmethod
    def explain(self, df, request_id, top_n=5) -> ExplanationResult:
        """Explain one prediction. df must contain exactly one row."""


class ShapTreeExplainer(BaseExplainer):
    """SHAP explanations for tree-based models.

    Construct ONCE at service startup. Building a TreeExplainer is far more
    expensive than using it, so building per request would dominate latency.
    """

    def __init__(self, pipeline, metadata, display_names=None):
        self.pipeline = pipeline
        self.metadata = metadata
        self.display_names = display_names or DISPLAY_NAMES

        feature_names = metadata['feature_names']
        missing = [f for f in feature_names if f not in self.display_names]
        if missing:
            raise MissingDisplayNameError(
                f"No display name for engineered feature(s): {missing}. "
                f"Add them to DISPLAY_NAMES in explain_service/explainer.py — "
                f"the LLM prompt depends on these labels."
            )

        self.explainer = shap.TreeExplainer(pipeline.model)

    # -- internal helpers ------------------------------------------------

    @staticmethod
    def _positive_class_values(raw):
        """Return the SHAP values for the positive class as a 1D array.

        Handles the shapes different SHAP/XGBoost versions produce:
          - list of arrays, one per class  -> take index 1
          - 3D array (n, features, classes) -> take [..., 1]
          - 2D array (n, features)          -> already positive class
        """
        if isinstance(raw, list):
            raw = raw[1] if len(raw) > 1 else raw[0]
        values = np.asarray(raw)
        if values.ndim == 3:
            values = values[..., 1] if values.shape[2] > 1 else values[..., 0]
        return values[0]

    @staticmethod
    def _scalar_base_value(raw):
        """Return the expected value as a float, whatever shape it arrives in."""
        if isinstance(raw, (list, tuple, np.ndarray)):
            arr = np.asarray(raw).ravel()
            return float(arr[1] if arr.size > 1 else arr[0])
        return float(raw)

    # -- public API ------------------------------------------------------

    def explain(self, df, request_id, top_n=5) -> ExplanationResult:
        if len(df) != 1:
            raise ValueError(
                f"explain() handles exactly one row, received {len(df)}. "
                f"For batches, call it per row."
            )

        cleaned = self.pipeline.clean(df)
        X, _ = self.pipeline.feature_engineering(cleaned, fit=False)

        shap_values = self._positive_class_values(self.explainer.shap_values(X))
        base_value = self._scalar_base_value(self.explainer.expected_value)

        probability = float(self.pipeline.model.predict_proba(X)[:, 1][0])

        all_contributions = []
        for i, feature in enumerate(X.columns):
            shap_value = float(shap_values[i])
            all_contributions.append(
                FeatureContribution(
                    feature=feature,
                    display_name=self.display_names[feature],
                    value=float(X.iloc[0, i]),
                    shap_value=shap_value,
                    direction='increases_risk' if shap_value > 0 else 'decreases_risk',
                    rank=0,
                )
            )

        ranked = sorted(all_contributions, key=lambda c: abs(c.shap_value), reverse=True)
        top = ranked[:top_n]
        for position, contribution in enumerate(top, start=1):
            contribution.rank = position

        return ExplanationResult(
            request_id=request_id,
            probability=probability,
            base_value=base_value,
            contributions=top,
            model_version=self.metadata['model_version'],
            pipeline_name=self.metadata['pipeline_name'],
            all_contributions=all_contributions,
        )