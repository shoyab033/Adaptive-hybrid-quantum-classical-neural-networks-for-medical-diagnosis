"""
Data splitting utilities.
Implements reproducible, stratified train/test and K-fold CV splitting.
"""

from typing import Generator, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def get_stratified_cv_splits(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
    shuffle: bool = True
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generate stratified K-fold indices.
    Yields (train_indices, val_indices) for each fold.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        yield train_idx, val_idx


def get_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform a single stratified holdout train/test split.
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
