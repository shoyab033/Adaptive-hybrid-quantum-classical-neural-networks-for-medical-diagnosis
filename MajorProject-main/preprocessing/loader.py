"""
Dataset loader module.
Loads raw data based on dataset configuration and saves local cached copies in data/raw/.
"""

import os
import io
import yaml
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import load_breast_cancer


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def get_dataset_config(dataset_name: str) -> dict:
    """Load the YAML configuration for a given dataset."""
    root = get_project_root()
    config_path = root / "configs" / f"{dataset_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_breast_cancer_raw(config: dict) -> pd.DataFrame:
    """
    Load Wisconsin Diagnostic Breast Cancer (WDBC) dataset.
    Uses sklearn.datasets.load_breast_cancer (which bundles the 569x30 UCI WDBC dataset).
    """
    root = get_project_root()
    raw_path = root / config["raw_data_path"]
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    if raw_path.exists():
        df = pd.read_csv(raw_path)
    else:
        cancer_data = load_breast_cancer(as_frame=True)
        df = cancer_data.frame.copy()
        # In sklearn, target is 0=malignant, 1=benign.
        # In medical literature and standard convention, Malignant=1 (positive/disease), Benign=0 (negative).
        # We rename columns to match config format:
        col_map = {col: col.replace(" ", "_") for col in df.columns if col != "target"}
        df = df.rename(columns=col_map)
        df["diagnosis"] = df["target"].apply(lambda x: 1 if x == 0 else 0)  # 0 (malignant in sklearn) -> 1
        df = df.drop(columns=["target"])
        df.to_csv(raw_path, index=False)

    return df


def load_heart_disease_raw(config: dict) -> pd.DataFrame:
    """
    Load UCI Heart Disease (Cleveland) dataset.
    Downloads from UCI ML Repository if not already cached locally.
    """
    root = get_project_root()
    raw_path = root / config["raw_data_path"]
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    if raw_path.exists():
        df = pd.read_csv(raw_path)
    else:
        url = config.get("raw_data_url", "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data")
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            content = response.text
        except Exception as e:
            # Fallback URL if primary UCI URL is slow or unreachable
            fallback_url = "https://raw.githubusercontent.com/alandelao/heart-disease-dataset/master/processed.cleveland.data"
            try:
                response = requests.get(fallback_url, timeout=15)
                response.raise_for_status()
                content = response.text
            except Exception as e2:
                raise RuntimeError(f"Failed to download Cleveland Heart dataset from {url} and fallback {fallback_url}: {e2}")

        df = pd.read_csv(io.StringIO(content), header=None, names=config["raw_columns"], na_values=["?"])
        df.to_csv(raw_path, index=False)

    return df


def load_diabetes_raw(config: dict) -> pd.DataFrame:
    """
    Load Pima Indians Diabetes Database.
    Downloads from authoritative repository if not already cached locally.
    """
    root = get_project_root()
    raw_path = root / config["raw_data_path"]
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    if raw_path.exists():
        df = pd.read_csv(raw_path)
    else:
        url = config.get("raw_data_url", "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv")
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            content = response.text
        except Exception as e:
            fallback_url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
            try:
                response = requests.get(fallback_url, timeout=15)
                response.raise_for_status()
                content = response.text
            except Exception as e2:
                raise RuntimeError(f"Failed to download Diabetes dataset from {url} and fallback: {e2}")

        df = pd.read_csv(io.StringIO(content), header=None, names=config["raw_columns"])
        df.to_csv(raw_path, index=False)

    return df


def load_dataset_by_name(dataset_name: str) -> tuple[pd.DataFrame, dict]:
    """
    Dispatcher to load any configured dataset by name.
    Returns: (raw_df, config_dict)
    """
    config = get_dataset_config(dataset_name)
    if dataset_name == "breast_cancer":
        df = load_breast_cancer_raw(config)
    elif dataset_name == "heart":
        df = load_heart_disease_raw(config)
    elif dataset_name == "diabetes":
        df = load_diabetes_raw(config)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Expected 'breast_cancer', 'heart', or 'diabetes'.")
    return df, config
