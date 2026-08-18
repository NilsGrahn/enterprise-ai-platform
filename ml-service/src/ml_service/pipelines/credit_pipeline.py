import numpy as np
import pandas as pd
from ml_service.base_pipeline import BasePipeline, TrainingResult
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
from xgboost import XGBClassifier


def ks_statistic(y_true, y_score) -> float:
    """Kolmogorov-Smirnov statistic: the largest gap between the true-positive
    and false-positive rates across all thresholds.

    In credit scoring this is the standard companion metric to AUC — it
    answers 'how well does this model separate defaulters from non-defaulters
    at its single best cut-off'.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


class CreditPipeline(BasePipeline):
    """Credit default risk: predicts P(serious delinquency within 2 years)."""

    name = 'credit'
    target_column = 'is_serious_delinquency'
    source_table = 'gold.v_credit_assessment'

    @property
    def required_columns(self):
        return [
            'revolving_utilisation',
            'age',
            'debt_ratio',
            'monthly_income',
            'open_credit_lines',
            'real_estate_loans',
            'times_30_59_days_late',
            'times_60_89_days_late',
            'times_90_days_late',
            'total_delinquency_events',
            'number_of_dependents',
        ]

    # ---- clean --------------------------------------------------------

    def clean(self, df):
        """Row-level validity only. No imputation."""
        df = df.copy()

        # Drop rows with no label. Only relevant at training time; at serve
        # time the target column is not present at all.
        if self.target_column in df.columns:
            df = df[df[self.target_column].notna()]

        for col in self.required_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Age outside a plausible lending range is a data error, not a signal.
        df['age'] = df['age'].clip(lower=18, upper=100)

        # Counts cannot be negative.
        count_columns = [
            'open_credit_lines',
            'real_estate_loans',
            'times_30_59_days_late',
            'times_60_89_days_late',
            'times_90_days_late',
            'total_delinquency_events',
            'number_of_dependents',
        ]
        for col in count_columns:
            df[col] = df[col].clip(lower=0)

        return df

    # ---- feature engineering ------------------------------------------

    def feature_engineering(self, df, fit: bool):
        """Build the model input matrix.

        fit=True  -> learn medians and caps from this data, return them
        fit=False -> apply the stored ones, learn nothing
        """
        if fit:
            p = {}
            p['median_income'] = float(df['monthly_income'].median())
            p['utilisation_cap'] = float(df['revolving_utilisation'].quantile(0.99))
            p['debt_ratio_cap'] = float(df['debt_ratio'].quantile(0.99))
        else:
            if self.preprocessing is None:
                raise ValueError(
                    "feature_engineering(fit=False) requires self.preprocessing "
                    "to be set. Load it from the artifact's metadata.json."
                )
            p = dict(self.preprocessing)

        X = pd.DataFrame(index=df.index)

        # Missingness is itself predictive here, so keep it as a feature
        # BEFORE filling the gap.
        X['income_missing'] = df['monthly_income'].isna().astype(int)

        X['monthly_income'] = df['monthly_income'].fillna(p['median_income'])
        X['revolving_utilisation'] = df['revolving_utilisation'].clip(upper=p['utilisation_cap'])
        X['debt_ratio'] = df['debt_ratio'].clip(upper=p['debt_ratio_cap'])

        X['age'] = df['age']
        X['number_of_dependents'] = df['number_of_dependents'].fillna(0)
        X['open_credit_lines'] = df['open_credit_lines'].fillna(0)
        X['real_estate_loans'] = df['real_estate_loans'].fillna(0)
        X['times_30_59_days_late'] = df['times_30_59_days_late'].fillna(0)
        X['times_60_89_days_late'] = df['times_60_89_days_late'].fillna(0)
        X['times_90_days_late'] = df['times_90_days_late'].fillna(0)
        X['total_delinquency_events'] = df['total_delinquency_events'].fillna(0)

        # ---- derived features ----
        X['monthly_debt_payment'] = X['debt_ratio'] * X['monthly_income']
        X['income_per_dependent'] = X['monthly_income'] / (X['number_of_dependents'] + 1)
        X['delinquency_ratio'] = X['total_delinquency_events'] / (X['open_credit_lines'] + 1)
        X['has_any_delinquency'] = (X['total_delinquency_events'] > 0).astype(int)
        X['utilisation_bucket'] = pd.cut(
            X['revolving_utilisation'],
            bins=[-0.01, 0.1, 0.3, 0.6, 1.0, np.inf],
            labels=False,
        ).astype(int)

        X['age'] = X['age'].fillna(X['age'].median() if fit else p.get('median_age', 45))
        if fit:
            p['median_age'] = float(X['age'].median())

        if fit:
            p['feature_names'] = list(X.columns)

        # Reindex to the frozen order. This is what makes serving safe.
        return X[p['feature_names']], p

    # ---- train --------------------------------------------------------

    def train(self, X_train, y_train, X_valid, y_valid) -> TrainingResult:
        n_negative = int((y_train == 0).sum())
        n_positive = int((y_train == 1).sum())
        scale_pos_weight = n_negative / n_positive

        print(f"class balance: {n_negative} negative / {n_positive} positive "
              f"-> scale_pos_weight={scale_pos_weight:.2f}")

        model = XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='auc',
            early_stopping_rounds=50,
            scale_pos_weight=scale_pos_weight,
            random_state=self.settings.random_seed,
            n_jobs=-1,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
        )

        probabilities = model.predict_proba(X_valid)[:, 1]
        threshold = self.settings.decision_threshold
        predictions = (probabilities >= threshold).astype(int)

        metrics = {
            'auc': float(roc_auc_score(y_valid, probabilities)),
            'ks': ks_statistic(y_valid, probabilities),
            'precision': float(precision_score(y_valid, predictions, zero_division=0)),
            'recall': float(recall_score(y_valid, predictions, zero_division=0)),
            'threshold': float(threshold),
            'best_iteration': int(getattr(model, 'best_iteration', 0) or 0),
        }

        return TrainingResult(
            model=model,
            metrics=metrics,
            feature_names=list(X_train.columns),
            preprocessing=self.preprocessing,
            training_rows=len(X_train),
        )

    # ---- predict ------------------------------------------------------

    def predict(self, df):
        """Raw records in, probability of default out."""
        if self.model is None:
            raise ValueError(
                "No model loaded. Either run run_training() or construct the "
                "pipeline from a saved artifact."
            )
        cleaned = self.clean(df)
        X, _ = self.feature_engineering(cleaned, fit=False)
        return self.model.predict_proba(X)[:, 1]