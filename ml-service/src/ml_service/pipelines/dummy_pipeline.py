"""A deliberately crude second pipeline.

Its only purpose is to prove the BasePipeline abstraction from ADR-0004 is
real: swapping ACTIVE_PIPELINE to 'dummy' must leave the API, dashboard,
monitoring, containers and CI working with no edits outside this folder.

It differs from CreditPipeline in every way the abstraction is supposed to
tolerate:
  - a different source table (the raw fact table, not the view)
  - two features instead of seventeen
  - a different algorithm (a decision tree, not gradient boosting)
  - almost no cleaning and no learned preprocessing
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier

from ml_service.base_pipeline import BasePipeline, TrainingResult


class DummyPipeline(BasePipeline):
    """Minimal viable pipeline. Not intended to be a good model."""

    name = 'dummy'
    target_column = 'is_serious_delinquency'
    source_table = 'gold.fact_credit_assessment'

    @property
    def required_columns(self):
        return ['revolving_utilisation', 'total_delinquency_events']

    # ---- clean ---------------------------------------------------------

    def clean(self, df):
        """Drop unlabelled rows and coerce to numeric. Nothing else."""
        df = df.copy()

        if self.target_column in df.columns:
            df = df[df[self.target_column].notna()]

        for column in self.required_columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')

        return df

    # ---- feature engineering -------------------------------------------

    def feature_engineering(self, df, fit: bool):
        """Two columns, nulls filled with zero. No learned parameters.

        Still honours the fit contract: fit=True produces the preprocessing
        dict, fit=False consumes the stored one and learns nothing.
        """
        if fit:
            p = {}
        else:
            if self.preprocessing is None:
                raise ValueError(
                    "feature_engineering(fit=False) requires self.preprocessing. "
                    "Load it from the artifact's metadata.json."
                )
            p = dict(self.preprocessing)

        X = pd.DataFrame(index=df.index)
        X['revolving_utilisation'] = df['revolving_utilisation'].fillna(0).clip(upper=2.0)
        X['total_delinquency_events'] = df['total_delinquency_events'].fillna(0)

        if fit:
            p['feature_names'] = list(X.columns)

        return X[p['feature_names']], p

    # ---- train ---------------------------------------------------------

    def train(self, X_train, y_train, X_valid, y_valid) -> TrainingResult:
        model = DecisionTreeClassifier(
            max_depth=4,
            min_samples_leaf=50,
            class_weight='balanced',
            random_state=self.settings.random_seed,
        )
        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_valid)[:, 1]
        threshold = self.settings.decision_threshold
        predictions = (probabilities >= threshold).astype(int)

        fpr, tpr, _ = roc_curve(y_valid, probabilities)

        metrics = {
            'auc': float(roc_auc_score(y_valid, probabilities)),
            'ks': float(np.max(tpr - fpr)),
            'precision': float(precision_score(y_valid, predictions, zero_division=0)),
            'recall': float(recall_score(y_valid, predictions, zero_division=0)),
            'threshold': float(threshold),
            'best_iteration': 0,
        }

        return TrainingResult(
            model=model,
            metrics=metrics,
            feature_names=list(X_train.columns),
            preprocessing=self.preprocessing,
            training_rows=len(X_train),
        )

    # ---- predict -------------------------------------------------------

    def predict(self, df):
        if self.model is None:
            raise ValueError(
                "No model loaded. Run run_training() or construct the pipeline "
                "from a saved artifact."
            )
        cleaned = self.clean(df)
        X, _ = self.feature_engineering(cleaned, fit=False)
        return self.model.predict_proba(X)[:, 1]