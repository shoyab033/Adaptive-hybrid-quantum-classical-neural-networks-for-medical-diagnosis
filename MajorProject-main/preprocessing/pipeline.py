"""
Leakage-safe preprocessing pipeline.
Fits imputers, encoders, and scalers ONLY on training folds, and transforms validation/test folds.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


class LeakageSafePipeline:
    """
    Leakage-safe preprocessor that fits exclusively on training data and transforms test/validation data.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.num_cols: List[str] = config["features"].get("numerical", [])
        self.cat_cols: List[str] = config["features"].get("categorical", [])
        self.fitted: bool = False
        self.preprocessor: Optional[ColumnTransformer] = None
        self.feature_names_out: List[str] = []

    def _build_preprocessor(self, X_train: pd.DataFrame) -> ColumnTransformer:
        transformers = []

        # Available numerical columns in X_train
        active_num_cols = [c for c in self.num_cols if c in X_train.columns]
        active_cat_cols = [c for c in self.cat_cols if c in X_train.columns]

        if active_num_cols:
            from sklearn.pipeline import Pipeline as SkPipeline
            num_pipeline = SkPipeline([
                ("imputer", SimpleImputer(strategy=self.config["missing_values"].get("imputation_strategy", "median"))),
                ("scaler", StandardScaler()),
            ])
            transformers.append(("num", num_pipeline, active_num_cols))

        if active_cat_cols:
            from sklearn.pipeline import Pipeline as SkPipeline
            cat_pipeline = SkPipeline([
                ("imputer", SimpleImputer(strategy=self.config["missing_values"].get("imputation_strategy_cat", "most_frequent"))),
                ("encoder", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
            ])
            transformers.append(("cat", cat_pipeline, active_cat_cols))

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop"
        )
        return preprocessor

    def fit(self, X_train: pd.DataFrame, y_train: Optional[pd.Series] = None) -> "LeakageSafePipeline":
        """Fit all preprocessing transformers exclusively on training data."""
        self.preprocessor = self._build_preprocessor(X_train)
        self.preprocessor.fit(X_train)
        self.fitted = True
        
        # Extract output feature names
        try:
            self.feature_names_out = list(self.preprocessor.get_feature_names_out())
        except Exception:
            self.feature_names_out = [f"feat_{i}" for i in range(self.transform(X_train).shape[1])]
            
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using transformers fitted on training data."""
        if not self.fitted or self.preprocessor is None:
            raise RuntimeError("Pipeline must be fitted before transforming data.")
        return self.preprocessor.transform(X).astype(np.float32)

    def fit_transform(self, X_train: pd.DataFrame, y_train: Optional[pd.Series] = None) -> np.ndarray:
        """Fit on training data and transform it."""
        return self.fit(X_train, y_train).transform(X_train)
