# AIHC 2027 REPRODUCIBILITY GUIDE & EXPERIMENT ARTIFACT PACKAGE

**Paper Title:** Adaptive Hybrid Quantum-Classical Neural Network for Early Medical Disease Diagnosis Using Explainable AI  
**Paper ID:** 516 | **Track:** 2 (ML in Clinical Decision Support and Predictive Analytics)  
**Conference:** IEEE Artificial Intelligence in Healthcare Conference (AIHC 2027)  

This guide provides end-to-end instructions for reproducing all experimental results, statistical hypothesis tests, explainability audits, and the camera-ready IEEE PDF for Paper #516.

---

## 1. System Environment and Dependencies

### Hardware Environment
- **Processor:** Intel 8-core CPU (tested on Intel Core i7 / Xeon)
- **RAM:** Minimum 8 GB (16 GB recommended)
- **GPU:** Optional (all variational circuits use exact state-vector simulation via PennyLane CPU backends)

### Software Environment
- **Operating System:** Windows 10/11 or Ubuntu Linux 22.04+
- **Python Version:** Python 3.10 to 3.13 (tested on Python 3.13.0)

### Key Python Packages
```text
torch>=2.6.0
pennylane>=0.45.1
pennylane-lightning>=0.45.0
scikit-learn>=1.7.0
numpy>=2.0.0
pandas>=2.2.0
scipy>=1.15.0
shap>=0.52.0
reportlab>=5.0.0
pypdf>=6.20.0
```

To install the exact required dependencies:
```powershell
python -m pip install torch pennylane pennylane-lightning scikit-learn numpy pandas scipy shap reportlab pypdf
```

---

## 2. Directory Layout and Key Artifacts

```
adaptive-hqnn/
├── camera_ready/
│   ├── CAMERA_READY_AUDIT.md            # Comprehensive initial codebase and reviewer audit
│   ├── CHANGELOG.md                     # Detailed change audit across revisions
│   ├── README_REPRODUCIBILITY.md        # This reproducibility manual
│   ├── REVIEWER_RESPONSE_MATRIX.md      # Item-by-item response matrix for all 8 reviewers
│   ├── CAMERA_READY_REVISED.pdf         # Verified camera-ready PDF (5 pages, <= 6 page limit)
│   ├── backup/                          # Preserved original submission files and repeat-1 baseline
│   │   ├── paper_original/              # paper.tex.orig and paper.pdf.orig
│   │   └── repeat1_baseline/            # Initial 1-repeat 5-fold baseline data
│   ├── results/                         # All verified CSV results
│   │   ├── repeated_cv_raw_folds.csv    # 300 completed outer folds (5 repeats x 5 folds x 4 models x 3 datasets)
│   │   ├── repeated_cv_summary.csv      # Mean +- SD, 95% CIs, and workload metrics
│   │   ├── repeated_cv_statistical_tests.csv           # Nadeau-Bengio corrected resampled t-tests
│   │   ├── repeated_cv_statistical_tests_multitest.csv # Benjamini-Hochberg FDR adjustments
│   │   ├── ablation_benchmarks.csv      # 25-fold ablation summary (Path-0, 4-D bottleneck, M4 param-matched)
│   │   ├── ablation_raw_folds.csv       # Raw fold metrics for all ablations
│   │   ├── shap_fold_stability.csv      # Pairwise Spearman rank correlations across outer folds
│   │   ├── simulator_timing_benchmarks.csv             # 100 timed CPU inference iterations per path
│   │   ├── calibration_and_detailed_holdout.csv        # ECE, Brier score, and holdout generalization
│   │   ├── m4_detailed_workload_metrics.csv            # M4 refinement block distributions
│   │   └── m3_per_path_accuracy_analysis.csv           # Per-path preliminary vs final diagnostic accuracy
│   └── scripts/                         # Standalone executable experiment scripts
│       ├── exp_repeated_cv.py           # 5-repeat x 5-fold cross-validation engine
│       ├── exp_ablations.py             # 25-fold ablation suite runner
│       ├── exp_shap_stability.py        # Post-hoc Kernel SHAP fold stability evaluator
│       ├── exp_simulator_timing.py      # Simulator CPU inference latency benchmark
│       ├── exp_m4_workload.py           # M4 classical refinement workload evaluator
│       ├── exp_per_path_analysis.py     # M3 per-path preliminary vs final accuracy
│       ├── exp_holdout_calibration.py   # Holdout ECE and Brier score evaluation
│       ├── analyze_repeated_cv.py       # Nadeau-Bengio & FDR statistical testing engine
│       └── build_camera_ready_pdf.py    # ReportLab 5-page IEEE camera-ready PDF builder
├── data/
│   ├── raw/                             # Original benchmark datasets
│   └── splits/                          # Fixed development/holdout index splits (Seed 42)
└── paper/
    ├── paper.tex                        # Camera-ready LaTeX manuscript
    └── paper.pdf                        # Synchronized camera-ready PDF
```

---

## 3. Data Integrity and Partitioning Protocols

1. **Fixed Holdout Split (Seed 42):**
   - **WDBC Breast Cancer:** Total $N=569$; Development split $N=455$ (80%); Frozen Holdout $N=114$ (20%).
   - **UCI Cleveland Heart:** Total $N=303$; Development split $N=242$ (80%); Frozen Holdout $N=61$ (20%).
   - **Pima Indians Diabetes:** Total $N=768$; Development split $N=614$ (80%); Frozen Holdout $N=154$ (20%).
   - *Protocol Guarantee:* The holdout partitions were never exposed to training, hyperparameter selection, or threshold tuning.
2. **Repeated Cross-Validation Seeds:**
   - 5 independent random seeds: `[42, 101, 202, 303, 404]`.
   - Each seed generates a stratified 5-fold split of the development partition ($5 \times 5 = 25$ outer folds per dataset).
3. **Inner Cross-Validation Threshold Grid:**
   - Within each outer fold, an inner 3-fold cross-validation performs a grid search over:
     $$\tau_{high} \in [0.75, 0.80, 0.85, 0.90, 0.95], \quad \tau_{low} \in [0.55, 0.60, 0.65, 0.70] \quad (\text{s.t. } \tau_{high} > \tau_{low})$$
   - Selection score: $\mathcal{S}(\tau_{high}, \tau_{low}) = \text{F1}_{inner\_oof} - 0.05 \cdot (\bar{L}_{Q, inner\_oof} / 2.0)$.
   - Outer validation folds and holdout sets are completely isolated from threshold selection.

---

## 4. Step-by-Step Reproduction Instructions

### Step 1: Run Repeated Cross-Validation (Task B1)
Executes 300 full model runs across 5 repeats of 5-fold CV:
```powershell
python camera_ready/scripts/exp_repeated_cv.py --repeats 5
```
*Outputs generated:*
- `camera_ready/results/repeated_cv_raw_folds.csv` (300 rows)
- `camera_ready/results/repeated_cv_summary.csv`

### Step 2: Run Statistical Hypothesis Tests and FDR Adjustments (Task B4)
Computes Nadeau–Bengio resampled $t$-tests ($J=25, df=24$) and Benjamini–Hochberg FDR adjustments:
```powershell
python camera_ready/scripts/analyze_repeated_cv.py
```
*Outputs generated:*
- `camera_ready/results/repeated_cv_statistical_tests.csv`
- `camera_ready/results/repeated_cv_statistical_tests_multitest.csv`

### Step 3: Run the 25-Fold Ablation Suite (Task B2)
Evaluates Path-0-only, Classical 4-D Bottleneck, and Parameter-Matched M4 across all 25 folds:
```powershell
python camera_ready/scripts/exp_ablations.py
```
*Outputs generated:*
- `camera_ready/results/ablation_raw_folds.csv`
- `camera_ready/results/ablation_benchmarks.csv`

### Step 4: Run the SHAP Fold-Stability Audit (Task B3)
Audits feature rank stability across all 5 outer-fold checkpoints:
```powershell
python camera_ready/scripts/exp_shap_stability.py
```
*Output generated:*
- `camera_ready/results/shap_fold_stability.csv`

### Step 5: Benchmark State-Vector Simulation Latencies (Task B5)
Measures inference execution latencies over 100 CPU iterations:
```powershell
python camera_ready/scripts/exp_simulator_timing.py
```
*Output generated:*
- `camera_ready/results/simulator_timing_benchmarks.csv`

### Step 6: Evaluate M4 Refinement Workloads (Task B6)
Computes M4 classical refinement block distributions:
```powershell
python camera_ready/scripts/exp_m4_workload.py
```
*Output generated:*
- `camera_ready/results/m4_detailed_workload_metrics.csv`

### Step 7: Analyze M3 Per-Path Accuracy and Holdout Calibration (Task B7)
Computes preliminary vs. final diagnostic accuracy per path and probability calibration:
```powershell
python camera_ready/scripts/exp_per_path_analysis.py
python camera_ready/scripts/exp_holdout_calibration.py
```
*Outputs generated:*
- `camera_ready/results/m3_per_path_accuracy_analysis.csv`
- `camera_ready/results/calibration_and_detailed_holdout.csv`

### Step 8: Compile the Camera-Ready PDF (Task D)
Generates the publication-ready 5-page IEEE conference PDF:
```powershell
python camera_ready/scripts/build_camera_ready_pdf.py
```
*Outputs generated:*
- `camera_ready/CAMERA_READY_REVISED.pdf`
- Synchronized to `paper/paper.pdf`

---

## 5. Manuscript Table to Source Data Mapping

| Manuscript Table | Description | Primary Result File | Relevant Columns / Fields |
| :--- | :--- | :--- | :--- |
| **TABLE I** | Dataset Characteristics | `data/splits/*/splits_config.json` | Sample counts, feature counts, class balance, missing percentages |
| **TABLE II** | Repeated-CV (25 Folds) and Holdout Performance | `camera_ready/results/repeated_cv_summary.csv`, `calibration_and_detailed_holdout.csv` | `acc_mean`, `acc_std`, `acc_ci95`, `f1_mean`, `f1_std`, `f1_ci95`, `auc_mean`, `test_f1`, `test_auc` |
| **TABLE III** | Inference Routing Allocation | `camera_ready/results/repeated_cv_summary.csv` | `route_p0_pct`, `route_p1_pct`, `route_p2_pct`, `avg_q_layers`, `quantum_savings_pct` |
| **TABLE IV** | 25-Fold Ablation Suite Benchmarks | `camera_ready/results/ablation_benchmarks.csv` | `params`, `f1_mean`, `f1_std`, `f1_ci95`, `auc_mean`, `auc_std` |
| **TABLE V** | Nadeau–Bengio Resampled $t$-Tests ($df=24$) | `camera_ready/results/repeated_cv_statistical_tests.csv`, `...multitest.csv` | `diff_mean`, `t_nb`, `p_nb`, `p_fdr`, `is_significant` |

---

## 6. Verification and Validation Checklist

- [x] All 300 repeated-CV runs completed without missing, duplicate, or divergent folds.
- [x] Fixed holdout test partition completely isolated from training and inner threshold tuning.
- [x] Nadeau–Bengio corrected resampled $t$-test properly applied ($df=24$, variance correction factor 0.29).
- [x] False Discovery Rate (FDR) controlled across all 27 pairwise tests via Benjamini–Hochberg.
- [x] Parameter counts calculated directly from PyTorch model graphs.
- [x] Physical quantum hardware and cryogenic claims strictly avoided.
- [x] Camera-ready PDF compiled to exactly 5 pages ($\le 6$ page limit) with zero orphan text.

