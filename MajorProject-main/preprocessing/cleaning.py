"""
Data cleaning and validation module.
Applies domain-specific medical data cleaning rules (e.g. physiological zero handling, target binarization).
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any


def clean_breast_cancer(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series]:
    """Clean Breast Cancer WDBC data."""
    df_clean = df.copy()
    target_col = config["target"]["name"]
    
    # Drop unnecessary columns if present (e.g., id)
    for col in config["features"].get("drop_columns", []):
        if col in df_clean.columns:
            df_clean = df_clean.drop(columns=[col])

    # Extract target
    y = df_clean[target_col].astype(int)
    X = df_clean.drop(columns=[target_col])
    
    return X, y


def clean_heart_disease(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Clean UCI Cleveland Heart Disease data:
    - Binarize target `num`: 0 -> 0 (no disease), >0 -> 1 (disease present)
    - Convert missing value markers ('?') to NaN
    """
    df_clean = df.copy()
    target_col = config["target"]["name"]

    # Handle missing markers
    token = config["missing_values"].get("token", "?")
    if token:
        df_clean = df_clean.replace(token, np.nan)

    # Convert object columns containing numbers with NaNs to float
    for col in df_clean.columns:
        if df_clean[col].dtype == object:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # Binarize target
    y_raw = df_clean[target_col].astype(float)
    y = (y_raw > 0).astype(int)
    X = df_clean.drop(columns=[target_col])

    return X, y


def clean_diabetes(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Clean Pima Indians Diabetes Database:
    - Treat physiologically implausible 0 values in Glucose, BloodPressure, SkinThickness, Insulin, BMI as NaN.
    - Note: Pregnancies and Outcome are legitimately 0.
    """
    df_clean = df.copy()
    target_col = config["target"]["name"]

    zero_as_missing = config["missing_values"].get("zero_as_missing_columns", [])
    for col in zero_as_missing:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].replace(0, np.nan)

    y = df_clean[target_col].astype(int)
    X = df_clean.drop(columns=[target_col])

    return X, y


def clean_dataset(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series]:
    """Dispatch dataset-specific cleaning."""
    dataset_name = config["dataset_name"]
    if dataset_name == "breast_cancer":
        return clean_breast_cancer(df, config)
    elif dataset_name == "heart":
        return clean_heart_disease(df, config)
    elif dataset_name == "diabetes":
        return clean_diabetes(df, config)
    else:
        raise ValueError(f"Unsupported dataset name for cleaning: {dataset_name}")
