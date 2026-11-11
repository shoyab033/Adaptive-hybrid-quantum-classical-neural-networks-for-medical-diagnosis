"""
Stage 1: Dataset Verification, Leakage-Safe Preprocessing Validation, and Summary Reporting.
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import yaml
from preprocessing.loader import load_dataset_by_name, get_dataset_config
from preprocessing.cleaning import clean_dataset
from preprocessing.pipeline import LeakageSafePipeline
from preprocessing.data_split import get_stratified_cv_splits, get_train_test_split


def inspect_and_validate_dataset(dataset_name: str) -> dict:
    """Inspect a single dataset, run leakage-safe CV validation, and return summary metrics."""
    print(f"\n{'='*70}")
    print(f"Inspecting Dataset: {dataset_name.upper()}")
    print(f"{'='*70}")

    # 1. Load raw data
    raw_df, config = load_dataset_by_name(dataset_name)
    raw_samples, raw_cols = raw_df.shape
    print(f"Source: {config.get('source')}")
    print(f"Raw shape: {raw_samples} samples x {raw_cols} columns")

    # 2. Clean data
    X, y = clean_dataset(raw_df, config)
    clean_samples, clean_features = X.shape
    print(f"Processed feature matrix X: {clean_samples} samples x {clean_features} features")
    print(f"Target distribution y: \n{y.value_counts(normalize=True).to_dict()} (Counts: {y.value_counts().to_dict()})")

    # 3. Missing value analysis
    missing_count = X.isna().sum().sum()
    missing_cols = X.columns[X.isna().any()].tolist()
    print(f"Missing entries detected: {missing_count} across columns: {missing_cols}")

    # 4. Leakage-safe Train/Test Split Validation
    X_train, X_test, y_train, y_test = get_train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline = LeakageSafePipeline(config)
    X_train_trans = pipeline.fit_transform(X_train, y_train)
    X_test_trans = pipeline.transform(X_test)

    # Sanity checks on holdout
    assert not np.isnan(X_train_trans).any(), "NaNs detected in X_train after preprocessing!"
    assert not np.isnan(X_test_trans).any(), "NaNs detected in X_test after preprocessing!"
    assert X_train_trans.shape[0] == len(y_train), "Train dimension mismatch!"
    assert X_test_trans.shape[0] == len(y_test), "Test dimension mismatch!"
    print(f"[PASSED] Holdout split check: Train={X_train_trans.shape}, Test={X_test_trans.shape}, Transformed features={X_train_trans.shape[1]}")

    # 5. Leakage-safe 5-Fold CV Validation
    print("Validating 5-Fold Stratified Cross-Validation isolation...")
    fold_stats = []
    for fold, (train_idx, val_idx) in enumerate(get_stratified_cv_splits(X, y, n_splits=5, random_state=42)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_val = X.iloc[val_idx], y.iloc[val_idx]

        fold_pipe = LeakageSafePipeline(config)
        X_tr_trans = fold_pipe.fit_transform(X_tr, y_tr)
        X_va_trans = fold_pipe.transform(X_va)

        # Confirm no NaNs and proper dimensions
        assert not np.isnan(X_tr_trans).any(), f"Fold {fold+1}: NaNs in training fold!"
        assert not np.isnan(X_va_trans).any(), f"Fold {fold+1}: NaNs in validation fold!"

        # Confirm scaler parameters are strictly derived from train fold
        # (e.g. check mean of training features is ~0.0)
        mean_tr = np.mean(X_tr_trans, axis=0)
        assert np.allclose(mean_tr[:len(config['features'].get('numerical', []))], 0, atol=1e-5), f"Fold {fold+1}: Scaler mean not zero on train!"

        pos_tr_ratio = y_tr.mean()
        pos_va_ratio = y_val.mean()
        fold_stats.append({
            "fold": fold + 1,
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "train_pos_rate": f"{pos_tr_ratio:.3f}",
            "val_pos_rate": f"{pos_va_ratio:.3f}",
        })

    print(f"[PASSED] 5-Fold Stratified CV verified with fold isolation.")
    for fs in fold_stats:
        print(f"  Fold {fs['fold']}: Train N={fs['train_samples']} (Pos={fs['train_pos_rate']}), Val N={fs['val_samples']} (Pos={fs['val_pos_rate']})")

    # Compile Table 1 Summary row
    class_0_cnt = int((y == 0).sum())
    class_1_cnt = int((y == 1).sum())
    class_dist_str = f"0: {class_0_cnt} ({class_0_cnt/len(y):.1%}) / 1: {class_1_cnt} ({class_1_cnt/len(y):.1%})"

    summary_info = {
        "Dataset": config["display_name"],
        "Source": config["source"],
        "Samples": clean_samples,
        "Raw Features": clean_features,
        "Transformed Features": X_train_trans.shape[1],
        "Missing Values": f"{missing_count} entries ({len(missing_cols)} columns)",
        "Class Distribution (0 / 1)": class_dist_str,
        "Target Meaning": config["target"].get("description", "0: Negative, 1: Positive"),
        "Imputation & Scaling": f"Impute: {config['missing_values'].get('imputation_strategy', 'median')}, Scale: Standard"
    }

    return summary_info


def main():
    print("=" * 80)
    print("STAGE 1: DATASET VERIFICATION & LEAKAGE-SAFE PIPELINE VALIDATION")
    print("=" * 80)

    datasets = ["breast_cancer", "heart", "diabetes"]
    summaries = []

    for ds in datasets:
        info = inspect_and_validate_dataset(ds)
        summaries.append(info)

    # Build Table 1 DataFrame
    table1_df = pd.DataFrame(summaries)
    
    # Save results
    results_dir = PROJECT_ROOT / "results" / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = results_dir / "table1_dataset_characteristics.csv"
    md_path = results_dir / "table1_dataset_characteristics.md"
    
    table1_df.to_csv(csv_path, index=False)
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Table 1: Dataset Characteristics and Preprocessing Specifications\n\n")
        f.write(table1_df.to_markdown(index=False))
        f.write("\n\n### Preprocessing Rules & Leakage-Safe Protocol Summary\n\n")
        f.write("1. **Breast Cancer (WDBC)**: 569 samples, 30 continuous features, zero missing values, standardized per training fold.\n")
        f.write("2. **UCI Cleveland Heart**: 303 samples, 13 features (5 numeric, 8 categorical), missing `?` tokens in `ca` (4) and `thal` (2) median/mode imputed inside training fold, one-hot encoded, and standardized.\n")
        f.write("3. **Pima Indians Diabetes**: 768 samples, 8 numerical features, physiological zeros in Glucose, BloodPressure, SkinThickness, Insulin, and BMI converted to NaN, median imputed strictly within training folds, and standardized.\n")
        f.write("4. **Isolation Guarantee**: All transformers are fit exclusively on training folds; validation and test splits are strictly out-of-sample.\n")

    print("\n" + "=" * 80)
    print("STAGE 1 VALIDATION COMPLETE: Table 1 Summary Generated")
    print("=" * 80)
    print(table1_df.to_string(index=False))
    print(f"\nSaved Table 1 CSV to: {csv_path}")
    print(f"Saved Table 1 Markdown to: {md_path}")


if __name__ == "__main__":
    main()
