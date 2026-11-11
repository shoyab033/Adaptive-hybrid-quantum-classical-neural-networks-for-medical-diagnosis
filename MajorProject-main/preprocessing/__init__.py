"""
Preprocessing package for Adaptive HQNN.
Provides configuration-driven, leakage-safe data loading, cleaning, and transformation pipelines.
"""

from .loader import load_dataset_by_name, get_dataset_config
from .cleaning import clean_dataset
from .pipeline import LeakageSafePipeline
from .data_split import get_stratified_cv_splits, get_train_test_split

__all__ = [
    "load_dataset_by_name",
    "get_dataset_config",
    "clean_dataset",
    "LeakageSafePipeline",
    "get_stratified_cv_splits",
    "get_train_test_split",
]
