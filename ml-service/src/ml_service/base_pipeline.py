from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd
from data_platform.db import get_engine
from sqlalchemy import text


class SchemaError(Exception):
    """Raised when input data is missing columns the pipeline requires."""


@dataclass
class TrainingResult:
    """Everything one training run produces.

    Returned by BasePipeline.train() and consumed by run_training().
    """
    model: Any
    metrics: dict
    feature_names: list
    preprocessing: dict
    training_rows: int


class BasePipeline(ABC):
    """Template for every domain pipeline.

    Infrastructure (train.py, evaluate.py, the API) only ever talks to this
    interface — never to a concrete subclass. Adding a new domain means
    writing a subclass and registering it; no infrastructure changes.
    """

    # ---- identity: every subclass MUST set these ----------------------
    name: str
    target_column: str
    source_table: str

    def __init__(self, settings=None, preprocessing: dict | None = None):
        from ml_service.config import get_settings

        self.settings = settings if settings is not None else get_settings()
        self.preprocessing = preprocessing
        self.model = None

    # ---- abstract: every subclass MUST implement these ----------------

    @property
    @abstractmethod
    def required_columns(self) -> list:
        """Columns that must be present in the input data."""

    @abstractmethod
    def clean(self, df):
        """Row-level validity only. Returns a DataFrame with the same columns.

        No imputation here — that is a modelling decision and belongs in
        feature_engineering, where it can be recorded in preprocessing.
        """

    @abstractmethod
    def feature_engineering(self, df, fit: bool):
        """Return (X, preprocessing).

        fit=True  -> learn parameters (medians, caps) and store them
        fit=False -> apply self.preprocessing, learn nothing
        """

    @abstractmethod
    def train(self, X_train, y_train, X_valid, y_valid) -> TrainingResult:
        """Fit a model and return it with its metrics."""

    @abstractmethod
    def predict(self, df):
        """Raw records in, probability of the positive class out."""

    # ---- concrete template methods: DO NOT override -------------------

    def load_data(self, split: str):
        """Load one dataset split from this pipeline's source table.

        Generic across domains because source_table is a class attribute.
        """
        engine = get_engine()
        query = text(f"SELECT * FROM {self.source_table} WHERE dataset_split = :split")
        return pd.read_sql(query, engine, params={'split': split})

    def validate_schema(self, df):
        """Raise SchemaError if any required column is absent."""
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            raise SchemaError(
                f"Pipeline '{self.name}' requires columns that are missing from "
                f"the input data: {missing}. "
                f"Present columns: {sorted(df.columns)}"
            )

    def run_training(self) -> TrainingResult:
        """The fixed training protocol. Order is not negotiable by subclasses."""
        from ml_service.artifacts import save_artifact
        from ml_service.registry import register_model

        # 1. load
        train_df = self.load_data('train')
        valid_df = self.load_data('valid')
        print(f"loaded {len(train_df)} train rows, {len(valid_df)} valid rows "
              f"from {self.source_table}")

        # 2. validate then clean
        self.validate_schema(train_df)
        self.validate_schema(valid_df)
        train_df = self.clean(train_df)
        valid_df = self.clean(valid_df)
        print(f"after cleaning: {len(train_df)} train, {len(valid_df)} valid")

        # 3. separate the target before engineering features
        y_train = train_df[self.target_column]
        y_valid = valid_df[self.target_column]

        # 4. engineer features — fit ONLY on train
        X_train, preprocessing = self.feature_engineering(train_df, fit=True)
        self.preprocessing = preprocessing
        X_valid, _ = self.feature_engineering(valid_df, fit=False)
        print(f"engineered {X_train.shape[1]} features")

        # 5. train
        result = self.train(X_train, y_train, X_valid, y_valid)
        self.model = result.model

        # 6. persist the artifact
        metadata = {
            'algorithm': type(result.model).__name__,
            'hyperparameters': result.model.get_params(),
            'training_rows': result.training_rows,
            'metrics': result.metrics,
            'preprocessing': result.preprocessing,
        }
        path = save_artifact(
            self.name,
            self.settings.model_version,
            result.model,
            metadata,
            result.feature_names,
        )
        print(f"saved artifact to {path}")

        # 7. register in the database
        model_key = register_model(
            pipeline_name=self.name,
            model_version=self.settings.model_version,
            algorithm=metadata['algorithm'],
            hyperparameters=metadata['hyperparameters'],
            feature_list=result.feature_names,
            training_rows=result.training_rows,
            metrics=result.metrics,
        )
        print(f"registered in gold.dim_model as model_key={model_key}")

        return result
    
    def build_feature_frame(self, df):
        """Return the engineered feature frame for raw input, without predicting.

        Used by the API to log exactly what the model consumed, and by the
        drift checker to describe the training distribution.
        """
        cleaned = self.clean(df)
        X, _ = self.feature_engineering(cleaned, fit=False)
        return X