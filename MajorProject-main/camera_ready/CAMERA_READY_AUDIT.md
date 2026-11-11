# CAMERA-READY AUDIT REPORT: AIHC 2027 (PAPER #516)

**Title:** Adaptive Hybrid Quantum-Classical Neural Network for Medical Diagnosis  
**Paper ID:** 516 | **Track:** 2 — ML in Clinical Decision Support and Predictive Analytics  
**Date of Audit:** October 31, 2026  
**Auditor:** Antigravity Research & Scientific Revision Assistant  
**Git Branch:** `camera-ready-revision` (Workspace initialized and protected)  
**Safety Backup:** `camera_ready/backup/paper/` and `camera_ready/backup/results/` (Preserved intact)

---

## 1. Files Inspected

The workspace, experimental artifacts, and reviewer materials were inspected across the following exact filesystem locations:

| Category | File Path | Status / Details |
| :--- | :--- | :--- |
| **Accepted Paper PDF** | `paper/paper.pdf` | Verified (6 pages, IEEE conference format, anonymous authors in tex, named in pdf) |
| **Editable Paper Source** | `paper/paper.tex`, `paper/paper.docx` | Verified (`IEEEtran` conference template, 341 lines) |
| **Reviewer Comments** | `C:\Users\yashw\Downloads\Conference Management Toolkit - View review.pdf` | Verified (13 pages, Reviewers #2, #4, #6, #7, #8, #9, #10, #11) |
| **Conference Template** | `C:\Users\yashw\Downloads\conference-latex-template_10-17-19.zip` | Verified (Official IEEE conference LaTeX package) |
| **Notebook 01** | `notebooks/01_dataset_validation_and_preprocessing.ipynb` | Verified (Data loading, median imputation, z-score scaling, 80/20 split) |
| **Notebook 02** | `notebooks/02_classical_mlp_baseline.ipynb` | Verified (Model 1 Classical MLP implementation and 5-fold CV) |
| **Notebook 03** | `notebooks/03_fixed_hqnn_baseline.ipynb` | Verified (Model 2 Fixed HQNN implementation and 5-fold CV) |
| **Notebook 04** | `notebooks/04_adaptive_hqnn.ipynb` | Verified (Model 3 Adaptive HQNN, nested inner CV, ensemble test) |
| **Notebook 05** | `notebooks/05_ablation_and_efficiency_benchmarks.ipynb` | Verified (Model 4 Adaptive Classical Control, Tables 2 & 3) |
| **Notebook 06** | `notebooks/06_xai_and_shap_analysis.ipynb` | Verified (Kernel SHAP analysis, patient waterfall plots, routing boxplots) |
| **Notebook 07** | `notebooks/07_statistical_evaluation_and_final_report.ipynb` | Verified (Paired Wilcoxon signed-rank tests, Cohen's d, Table 5) |
| **Data Splits** | `data/splits/{breast_cancer, heart, diabetes}/splits_config.json` | Verified (Fixed train/dev/test index splits, random seed 42) |
| **Raw Datasets** | `data/raw/{breast_cancer/wdbc.data, heart/processed.cleveland.data, diabetes/diabetes.csv}` | Verified (Integrity checksums match `dataset_sources_checksums.csv`) |
| **Trained Checkpoints** | `results/models/{classical_mlp, fixed_hqnn, adaptive_hqnn}/` | Verified (15 checkpoints per model family across 3 datasets x 5 folds) |
| **Evaluation Metrics** | `results/metrics/{m1, m2, m3, m4}_*.csv` | Verified (Per-fold and test ensemble metrics for all 4 models) |
| **Tables & Figures** | `results/tables/` (Tables 1-6) and `results/figures/` (17 figures) | Verified (Markdown, CSV, TeX, and PNG formats) |

---

## 2. Exact Verified Model Architectures

The architectures of all four benchmark models were inspected directly from the Python/PyTorch definitions:

### M1: Classical MLP (`ClassicalMLP`)
* **Layer Hierarchy:**
  $$\text{Input}(D) \xrightarrow{\text{Linear}} \text{Dense}(32) \xrightarrow{\text{LayerNorm}} \text{LayerNorm}(32) \xrightarrow{} \text{ReLU} \xrightarrow{} \text{Dropout}(p=0.2) \xrightarrow{\text{Linear}} \text{Dense}(16) \xrightarrow{} \text{ReLU} \xrightarrow{\text{Linear}} \text{Out}(1)$$
* **Layer Dimensions:** $D \to 32 \to 16 \to 1$
* **Activations & Normalization:** LayerNorm(32), ReLU, Dropout(0.2), ReLU. Output is raw logit.
* *(Note conflict with paper text: paper claimed $D \to 16 \to 8 \to 1$ with Dropout 0.1).*

### M2: Fixed HQNN (`FixedHQNN`)
* **Classical Feature Encoder:**
  $$\text{Input}(D) \xrightarrow{\text{Linear}} \text{Dense}(16) \xrightarrow{\text{LayerNorm}} \text{LayerNorm}(16) \xrightarrow{} \text{ReLU} \xrightarrow{\text{Linear}} \text{Dense}(4) \xrightarrow{} \text{Tanh} \xrightarrow{} \times \pi$$
* **Quantum Variational Circuit ($L_Q = 2.0$ fixed):**
  * Qubits: $Q = 4$
  * State preparation: $R_Y$ angle embedding of 4-d latent vector.
  * Layer 1: 4 parameterized Euler rotations `qml.Rot` $(\alpha_{1,j}, \beta_{1,j}, \gamma_{1,j})$ + circular CNOT entanglement ring $[(0,1), (1,2), (2,3), (3,0)]$.
  * Layer 2: 4 parameterized Euler rotations `qml.Rot` $(\alpha_{2,j}, \beta_{2,j}, \gamma_{2,j})$ + circular CNOT entanglement ring.
  * Readout: 4 Pauli-$Z$ expectation values $\langle Z_j \rangle \in [-1, 1]$.
* **Classification Readout:** $\text{Linear}(4, 1) \to \text{logit}$.

### M3: Adaptive HQNN (`AdaptiveHQNN`) — Proposed
* **Classical Feature Encoder (Shared across all paths):**
  $$h(x) = \text{ReLU}(\text{LayerNorm}(\text{Linear}(D, 16)))$$
* **Preliminary Classification Head (Shared path decider):**
  $$z_{prelim}(x) = \text{Linear}(16, 1) \to p_{prelim}(x) = \sigma(z_{prelim}(x)), \quad C(x) = \max(p_{prelim}, 1 - p_{prelim})$$
* **Latent Quantum Projection (Shared across quantum paths):**
  $$u(x) = \pi \cdot \tanh(\text{Linear}(16, 4)) \in [-\pi, \pi]^4$$
* **Path 0 ($L_Q = 0$):** High confidence ($C(x) \ge \tau_{high}$). Bypasses quantum execution entirely. Final logit is $z_{prelim}(x)$.
* **Path 1 ($L_Q = 1$):** Intermediate confidence ($\tau_{low} \le C(x) < \tau_{high}$).
  * Runs 1 variational layer: 4 `qml.Rot` + 4 circular CNOTs.
  * Readout: Pauli-$Z$ expectations $\to \text{Linear}_{q1}(4, 1) \to z_{q1}(x)$.
* **Path 2 ($L_Q = 2$):** Low confidence ($C(x) < \tau_{low}$).
  * Runs 2 variational layers: 8 `qml.Rot` + 8 circular CNOTs.
  * Readout: Pauli-$Z$ expectations $\to \text{Linear}_{q2}(4, 1) \to z_{q2}(x)$.

### M4: Adaptive Classical Control (`AdaptiveClassicalControl`)
* **Classical Feature Encoder (Shared):** $\text{Linear}(D, 16) \to \text{LayerNorm}(16) \to \text{ReLU}()$.
* **Preliminary Head (Shared):** $\text{Linear}(16, 1) \to z_{prelim}(x)$.
* **Classical Refinement 1 (Path 1 substitute):** $\text{Linear}(16, 16) \to \text{ReLU} \to \text{Linear}(16, 1)$ (produces additive residual logit: $z_{prelim} + \Delta z_{c1}$).
* **Classical Refinement 2 (Path 2 substitute):** $\text{Linear}(16, 32) \to \text{ReLU} \to \text{Dropout}(0.2) \to \text{Linear}(32, 16) \to \text{ReLU} \to \text{Linear}(16, 1)$ (produces additive residual logit: $z_{prelim} + \Delta z_{c2}$).
* **Dynamic Routing:** Same threshold logic on $C(x)$ with $\tau_{high}, \tau_{low}$. Bypasses refinement blocks when $C(x) \ge \tau_{high}$.

---

## 3. Parameter Counts and Calculation Method

Parameter counts were computed directly using PyTorch:  
`sum(p.numel() for p in model.parameters() if p.requires_grad)`

| Dataset | Input Dim ($D$) | M1 (Classical MLP) | M2 (Fixed HQNN) | M3 (Adaptive HQNN) | M4 (Adaptive Classical Control) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **WDBC Breast Cancer** | 30 | **1,601** | **625** (601 class. + 24 quant.) | **659** (623 class. + 36 quant.) | **1,923** |
| **UCI Heart Disease** | 13 | **1,057** | **353** (329 class. + 24 quant.) | **387** (351 class. + 36 quant.) | **1,651** |
| **Pima Indians Diabetes** | 8 | **897** | **273** (249 class. + 24 quant.) | **307** (271 class. + 36 quant.) | **1,571** |

### Breakdown of M3 Parameters ($D=30$):
* Encoder: `Linear(30, 16)` [480 weights + 16 bias] + `LayerNorm(16)` [16 weights + 16 bias] = 528 params
* Preliminary Head: `Linear(16, 1)` [16 weights + 1 bias] = 17 params
* Latent Projection: `Linear(16, 4)` [64 weights + 4 bias] = 68 params
* Path 1 Quantum Circuit: $1 \text{ layer} \times 4 \text{ qubits} \times 3 \text{ Euler angles} = 12$ trainable quantum parameters
* Path 2 Quantum Circuit: $2 \text{ layers} \times 4 \text{ qubits} \times 3 \text{ Euler angles} = 24$ trainable quantum parameters
* Path 1 Readout: `Linear(4, 1)` [4 weights + 1 bias] = 5 params
* Path 2 Readout: `Linear(4, 1)` [4 weights + 1 bias] = 5 params
* **Total M3 Parameters:** $528 + 17 + 68 + 12 + 24 + 5 + 5 = 659$ (623 classical, 36 quantum).

*(Critical Finding: M4 has nearly 3x the parameters of M3, contradicting the claim in paper lines 59 & 219 that M4 has "matching parametric capacity").*

---

## 4. Explanation of M4 (Adaptive Classical Control)

* **Architecture:** Replaces the quantum projection and VQC with classical multilayer residual refinement blocks (`c1_refine` and `c2_refine`).
* **Skipped Computation:** Confident samples ($C(x) \ge \tau_{high}$) exit early after the preliminary head, completely skipping both classical refinement blocks. Intermediate samples execute only `c1_refine` and skip `c2_refine`. Low-confidence samples execute `c2_refine`.
* **Quantum Circuits:** Uses zero quantum circuits ($L_Q = 0$).
* **Path Distribution & Workload:** In the original notebook `05_ablation_and_efficiency_benchmarks.ipynb`, `routes_val` was generated but discarded with `_`, and `avg_quantum_layers = 0.00` was hardcoded. M4's routing percentages (0-Refine %, 1-Refine %, 2-Refine %) and FLOP savings were **never recorded in `m4_adaptive_classical_metrics.csv` or published in Table II / Table 3**.
* **Remedy Needed:** Evaluate M4 checkpoints on the CV and test splits to compute its exact path distributions and FLOP workloads.

---

## 5. Threshold-Selection Audit

* **Confidence Definition:**  
  $$C(x) = \max(p_{prelim}(x), 1 - p_{prelim}(x)) \in [0.5, 1.0]$$
* **Threshold Selection Objective:**
  $$\text{Score}(\tau_{high}, \tau_{low}) = \text{F1}_{inner\_oof}(\tau_{high}, \tau_{low}) - 0.05 \cdot \left(\frac{\bar{L}_{Q, inner\_oof}(\tau_{high}, \tau_{low})}{2.0}\right)$$
  Evaluated across out-of-fold predictions from a 3-fold inner CV.
* **Candidate Grid:**
  * $\tau_{high} \in [0.75, 0.80, 0.85, 0.90, 0.95]$
  * $\tau_{low} \in [0.55, 0.60, 0.65, 0.70]$
  * Constraint: $\tau_{high} > \tau_{low}$ (19 valid candidate pairs).
* **Selected Thresholds Per Outer Fold (Verified from `m3_adaptive_hqnn_metrics.csv`):**
  * **WDBC Breast Cancer:**
    * Fold 1: $\tau_h = 0.90, \tau_l = 0.60$
    * Fold 2: $\tau_h = 0.75, \tau_l = 0.55$
    * Fold 3: $\tau_h = 0.80, \tau_l = 0.55$
    * Fold 4: $\tau_h = 0.75, \tau_l = 0.55$
    * Fold 5: $\tau_h = 0.85, \tau_l = 0.70$
    * *Holdout Test Ensemble:* Median thresholds $(\tau_h = 0.80, \tau_l = 0.55)$
  * **UCI Heart Disease:**
    * Fold 1: $\tau_h = 0.85, \tau_l = 0.55$
    * Fold 2: $\tau_h = 0.75, \tau_l = 0.55$
    * Fold 3: $\tau_h = 0.80, \tau_l = 0.70$
    * Fold 4: $\tau_h = 0.75, \tau_l = 0.55$
    * Fold 5: $\tau_h = 0.75, \tau_l = 0.55$
    * *Holdout Test Ensemble:* Median thresholds $(\tau_h = 0.75, \tau_l = 0.55)$
  * **Pima Indians Diabetes:**
    * Fold 1: $\tau_h = 0.75, \tau_l = 0.55$
    * Fold 2: $\tau_h = 0.75, \tau_l = 0.55$
    * Fold 3: $\tau_h = 0.75, \tau_l = 0.65$
    * Fold 4: $\tau_h = 0.75, \tau_l = 0.70$
    * Fold 5: $\tau_h = 0.75, \tau_l = 0.55$
    * *Holdout Test Ensemble:* Median thresholds $(\tau_h = 0.75, \tau_l = 0.55)$
* **Leakage Control:** Nested 3-fold inner cross-validation is performed strictly on the outer training partition ($N_{train} \approx 364$ for WDBC, $193$ for Heart, $491$ for Diabetes). Outer validation and holdout test partitions are never seen during threshold search.
* *(Note conflict with paper text: paper lines 173-174 claimed thresholds were averaged across folds, but code uses the median).*

---

## 6. Training Procedure

* **Preliminary Head & Quantum Heads:** Trained **jointly** on all training samples in every batch via `forward_all_paths(batch_x)`. All paths receive all training samples simultaneously.
* **Loss Function:** Composite binary cross-entropy with positive-class weighting:
  $$\mathcal{L}_{total} = \mathcal{L}_{BCE}(z_{prelim}, y) + 0.6 \cdot \mathcal{L}_{BCE}(z_{q1}, y) + 0.6 \cdot \mathcal{L}_{BCE}(z_{q2}, y)$$
  where $\mathcal{L}_{BCE}$ uses `pos_weight = float(n_neg / n_pos)` to handle class imbalance.
  *(Note conflict with paper Eq. 12: paper stated unweighted sum $1.0 + 1.0 + 1.0$ without mention of positive class weighting).*
* **Optimizer:** Adam with learning rate $\eta = 0.008$ (not $0.005$), `weight_decay = 1e-4`.
* **Scheduler:** `ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)`.
* **Early Stopping:** Monitored on inner validation loss with `patience=10-12` epochs (max epochs = 55–60).

---

## 7. Holdout Evaluation Method

* **Model Used for Holdout Set:** A **5-fold ensemble**.  
  All 5 outer fold models are evaluated on the untouched holdout test partition ($X_{test}$).  
  Ensemble preliminary, Q1, and Q2 probabilities are averaged:
  $$\bar{p}_{prelim} = \frac{1}{5} \sum_{k=1}^5 \sigma(z_{prelim}^{(k)}), \quad \bar{p}_{q1} = \frac{1}{5} \sum_{k=1}^5 \sigma(z_{q1}^{(k)}), \quad \bar{p}_{q2} = \frac{1}{5} \sum_{k=1}^5 \sigma(z_{q2}^{(k)})$$
  Ensemble confidence is $\bar{C}(x) = \max(\bar{p}_{prelim}, 1 - \bar{p}_{prelim})$.  
  Routing assignments on the holdout set use the median thresholds $(\tau_{high}^{med}, \tau_{low}^{med})$:
  $$\text{Route}(x) = \begin{cases} 0 & \bar{C}(x) \ge \tau_{high}^{med} \\ 1 & \tau_{low}^{med} \le \bar{C}(x) < \tau_{high}^{med} \\ 2 & \bar{C}(x) < \tau_{low}^{med} \end{cases}$$
* **Holdout Sample Counts Verified:**
  * **WDBC Breast Cancer:** Exactly **114 samples** (42 malignant, 72 benign; 20.04% of 569).
  * **UCI Cleveland Heart:** Exactly **61 samples** (28 disease, 33 normal; 20.13% of 303).
  * **Pima Indians Diabetes:** Exactly **154 samples** (54 diabetic, 100 non-diabetic; 20.05% of 768).

---

## 8. Quantum Circuit Audit

* **Initial State $|\psi_0\rangle$:** Computational ground state $|0\rangle^{\otimes 4} = |0000\rangle$ on PennyLane `default.qubit`.
* **Angle Embedding:** Single-qubit Pauli-$Y$ rotations:
  $$|\psi_0(u)\rangle = \bigotimes_{j=1}^4 R_Y(u_j) |0\rangle, \quad R_Y(\phi) = \exp\left(-i \frac{\phi}{2} \sigma_y\right)$$
* **Definition of $R(\theta)$ (Rot Gate):** Arbitrary single-qubit Euler rotation parameterized by 3 angles:
  $$R(\alpha, \beta, \gamma) = R_Z(\gamma) R_Y(\beta) R_Z(\alpha) = \exp\left(-i \frac{\gamma}{2} \sigma_z\right) \exp\left(-i \frac{\beta}{2} \sigma_y\right) \exp\left(-i \frac{\alpha}{2} \sigma_z\right)$$
  *(Each variational layer has $4 \times 3 = 12$ trainable rotation weights).*
* **Entanglement Pattern:** Circular ring of 4 CNOT gates: $(0 \to 1), (1 \to 2), (2 \to 3), (3 \to 0)$.
* **Measurement Operators:** Pauli-$Z$ expectation values $\langle Z_j \rangle$ across all 4 qubits $j \in \{0, 1, 2, 3\}$.
* **Exact Quantum Resource Specs (Derived from PennyLane `qml.specs`):**
  * **Path 1 (1 Q-Layer):**
    * Qubits: 4
    * Circuit Depth: **6**
    * High-level operations: 1 AngleEmbedding, 4 `Rot`, 4 CNOT (Total: 9 operations)
    * Elementary gates: 4 $R_Y$ + 12 single-qubit rotations + 4 CNOT = 20 gates
    * Measurements: 4 Pauli-$Z$ expectations
  * **Path 2 (2 Q-Layers):**
    * Qubits: 4
    * Circuit Depth: **11**
    * High-level operations: 1 AngleEmbedding, 8 `Rot`, 8 CNOT (Total: 17 operations)
    * Elementary gates: 4 $R_Y$ + 24 single-qubit rotations + 8 CNOT = 36 gates
    * Measurements: 4 Pauli-$Z$ expectations
* **Circuit Evaluation Count:** NOT tracked or logged during training or inference in the original notebooks.

---

## 9. Explainable AI (SHAP) Audit

* **Model Explained:** Adaptive HQNN (M3), **Fold 1 model checkpoint only** (`adaptive_hqnn_{ds}_fold_1.pt`).
* **Prediction Output Explained:** Final adaptive prediction probability after routing: `model_predict_proba` returns $[1 - p_{final}, p_{final}]$.
* **Kernel SHAP Settings:**
  * Background sample size: $N_{bg} = 30$ randomly sampled from development set.
  * Explained test sample size: $N_{eval} = 35$ test samples.
  * Number of perturbation evaluations: `nsamples = 65`.
  * Thresholds used: Hardcoded $\tau_{high} = 0.85, \tau_{low} = 0.65$ (not the fold-specific or ensemble-tuned thresholds).
* **Feature Ranking Procedure:** Mean absolute SHAP value across samples: $\frac{1}{N_{eval}} \sum_{i=1}^{N_{eval}} |\phi_{i,j}|$.
* **Fold-to-Fold Ranking Stability:** **NOT calculated** in the original experiments.

---

## 10. Holdout Sample Counts and Ground Truth

| Dataset | Total Samples ($N$) | Dev Samples (80%) | Holdout Samples (20%) | Holdout Positive Class | Holdout Negative Class |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **WDBC Breast Cancer** | 569 | 455 | **114** | 42 (Malignant, 36.8%) | 72 (Benign, 63.2%) |
| **UCI Heart Disease** | 303 | 242 | **61** | 28 (Disease, 45.9%) | 33 (Normal, 54.1%) |
| **Pima Indians Diabetes** | 768 | 614 | **154** | 54 (Diabetic, 35.1%) | 100 (Non-diabetic, 64.9%) |

---

## 11. Reviewer-by-Reviewer Audit Table

| Reviewer Comment | Paper Section | Required Action | Evidence Available | New Experiment Required? | Priority |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **R2.1:** No demonstrated quantum benefit; M1 beats M3; missing ablations (Path-0-only, classical 4-d bottleneck, parameter-matched MLP) | Sec. V-A, V-E, VI | Honestly report trade-off; avoid claiming quantum superiority; run ablations | M1 beats M3 in CV across all datasets | **Yes (Run Path-0 and classical bottleneck ablations)** | High |
| **R2.2:** M4 baseline not analysed, architecture & parameter count not given, workload absent from Table II | Sec. III, IV-D, Table II, Sec. V-E | Add full M4 description, parameter counts, and routing/workload stats | M4 defined in code; metrics saved in `m4_adaptive_classical_metrics.csv` | **Yes (Re-evaluate M4 for exact routing & workload)** | High |
| **R2.3:** Statistical testing invalid (5 folds cannot reach $p < 0.05$; $p=0.0625$ means Fixed beat Adaptive in all 5 folds); need repeated CV | Sec. V-C, VI | Remove claim of "no significant difference"; explain $N=5$ power limits; run repeated CV with CIs | Table 5 shows $W=0, p=0.0625$ on Heart & Diabetes | **Yes (Run repeated stratified CV, e.g., 5-10 repeats)** | Critical |
| **R2.4:** Holdout results over-read; small test sets; Table I omits test Acc, Prec, Rec; unclear holdout model protocol | Sec. IV-F, Table I, Sec. V-A | Clarify 5-fold ensemble protocol; tone down test claims; add test Acc, Prec, Rec to tables | Full test metrics verified in metrics CSVs | No (Data exists in CSVs) | High |
| **R2.5:** Workload is weak proxy; does not measure latency, energy, or noise; Path 1 vs 2 differs by only 4 rotations & 4 CNOTs | Sec. III-G, IV-F, V-B | Explicitly frame as proxy; report gate counts & simulator execution timings per path | Circuit gate counts derived (9 vs 17 ops); timing logs | **Yes (Measure inference time per path)** | Medium |
| **R2.6:** Under-specified training & capacity issue (4-d vs 16-d features); threshold objective & grid unstated; uncalibrated confidence | Sec. III-B, III-D, III-F, IV-E | State exact loss weights ($1.0, 0.6, 0.6$), pos_weight, threshold grid, inner CV objective, report per-path accuracy | Full implementation verified in notebook cells | **Yes (Compute preliminary vs final accuracy per path)** | High |
| **R2.7:** Baselines limited (only 1 MLP, no tree models), fixed hyperparameters, single-source small cohorts | Sec. IV-A, IV-D, VI | Acknowledge cohort limitations; document fixed hyperparameter protocol in Limitations | Full dataset and split details available | No (Document in text) | Medium |
| **R2.8:** XAI component thin; single paragraph, no figure; model & sample counts unstated; features classical | Sec. V-D, Fig. 5-7 | Add SHAP figure, state exact sample sizes ($N_{bg}=30, N_{eval}=35$), discuss top features carefully | Existing figures in `results/figures/fig_stage6_*` | **Yes (Calculate fold-to-fold ranking stability)** | High |
| **R2.9:** Outdated refs, missing early-exit/selective prediction literature, unspecified $R(\theta)$, undefined $|\psi_0\rangle$ | Sec. I, II, III-D, Refs | Define $|\psi_0\rangle$, $R(\alpha,\beta,\gamma)$ Euler decomposition; cite 2022-2026 early-exit and QML papers | Full gate definitions verified in PennyLane code | No (Update text & bibliography) | High |
| **R4.1:** Citations must follow IEEE numeric appearance order | References | Re-order all citations strictly by appearance order | Current `.bib`/`.tex` has unordered numeric keys | No (Update LaTeX) | High |
| **R4.2:** Four algorithms mentioned but not comparatively discussed | Sec. V-A, Table I | Add comparative discussion of M1, M2, M3, M4 across all metrics | Full metrics available in `table2_model_performance_comparison.csv` | No (Update text) | Medium |
| **R4.3:** XAI analysis needs detail and visual presentation | Sec. V-D | Include SHAP summary figure and explain clinical feature influences | Visuals available in `results/figures/` | No (Incorporate figure) | Medium |
| **R4.4:** References outdated (latest 2021); need 2022–2026 work | Sec. II, Refs | Add recent verified 2022-2026 citations on dynamic NNs and QML | Verified bibliography needed | No (Literature update) | High |
| **R4.5 & R4.6:** Define QL, CV, and all abbreviations at first use | Throughout | Define QL (Quantum Layer), CV (Cross-Validation), MLP, HQNN, VQC | Text audit | No (Text editing) | High |
| **R6.1 & R6.2:** Originality vs classical conditional computation; validate workload beyond layer counts | Sec. I, II-C, III-G | Clarify specific contribution (varying VQC depth based on classical confidence); add simulator latency | Circuit specs verified | **Yes (Simulator latency benchmark)** | Medium |
| **R6.3 & R6.4:** Trade-off analysis; explain when savings justify F1 reduction; threshold optimization details | Sec. III-F, V-C | State exact threshold objective equation and candidate grid; analyze operating trade-off | Full equations verified from notebook 04 | No (Add to Section III-F) | High |
| **R6.5 - R6.7:** Statistical power of 5-fold Wilcoxon; M4 comparison; SHAP ranking stability across folds | Sec. V-C, V-D | Replace 5-fold claim with repeated CV; analyse M4; report SHAP stability | Partial data exists | **Yes (Repeated CV & SHAP stability)** | High |
| **R7.1 - R7.5:** Novelty distinction, training details, simulator vs hardware, small dataset limits | Sec. II, III, IV, VI | Thoroughly document all methodology gaps and limitations | Verified in audit | No (Text revision) | Medium |
| **R8.1 - R8.2:** Limited datasets; report sensitivity, specificity, false-negative rates, calibration | Sec. IV-F, V-A, VI | Report precision, recall, specificity, confusion matrix counts; discuss calibration | Exact counts (TP, FP, TN, FN, Spec) verified in CSVs | No (Add to tables) | Medium |
| **R9.1 - R9.3:** Classical bottleneck risk (overconfident errors); clinical degradation danger; simulation limits | Sec. V-E, VI | Explicitly analyze risk of early-exit misclassification; disclaim clinical readiness | Data confirms F1 drop | No (Strengthen discussion) | High |
| **R10.1 - R10.6:** Runtime overhead of router; trade-off criteria; repeated random seeds; gate counts | Sec. III-D, IV-E, V-B, V-C | Add gate counts (9 vs 17), circuit depth (6 vs 11); run repeated seeds | Gate specs verified | **Yes (Repeated seeds, latency)** | High |
| **R11.1 - R11.7:** Technical distinctiveness; workload proxy; parameter counts; class imbalance handling | Sec. III, IV, Table I, II | Report exact parameter counts; describe `pos_weight` in BCE loss; add M4 workload | Verified parameter counts | No (Document in text & tables) | High |

---

## 12. Missing Information (`MISSING INFO`)

The following items are **not available** in existing saved outputs and require computation or author clarification:

1. **Simulator Inference Latency per Path:** Wall-clock inference runtime per sample for Path 0, Path 1, and Path 2 was not benchmarked.
2. **M4 Path Distribution & Workload:** The exact breakdown of samples routed to 0, 1, or 2 classical refinement blocks in M4 was not saved.
3. **Repeated Cross-Validation (Multiple Seeds):** Only a single 5-fold CV run (seed 42) was conducted. Reviewers 2, 6, 7, 10, 11 unanimously requested repeated CV (e.g., 5 or 10 repeats) with standard deviations and confidence intervals.
4. **Per-Path Preliminary vs. Final Accuracy for M3:** Accuracy of preliminary predictions specifically for samples escalated to Path 1 and Path 2 was not recorded.
5. **SHAP Fold-to-Fold Ranking Stability:** SHAP was computed only on Fold 1, so Spearman rank correlation across folds is currently absent.
6. **Probability Calibration Analysis:** Expected Calibration Error (ECE) or Brier scores for the preliminary confidence metric were not computed.
7. **Path-0-Only Baseline & Classical Bottleneck Ablation:** Proposed by Reviewer #2 to isolate whether the quantum layers contribute beyond the 4-d latent bottleneck.

---

## 13. Conflicts Between Paper, Code, and Outputs

| # | Topic | Statement in Accepted Paper | Actual Implementation in Code | Implication / Correction Required |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **M1 Architecture** | $D \to 16 \to 8 \to 1$ with Dropout(0.1) | $D \to 32 \to 16 \to 1$ with Dropout(0.2) and LayerNorm(32) | Update text to match actual code ($D \to 32 \to 16 \to 1$). |
| **2** | **Training Hyperparameters** | Adam with $\eta = 0.005$, batch size 32, 60 epochs for all models | M1: $\eta=0.003, \text{epochs}=150$; M2: $\eta=0.01, \text{epochs}=60$; M3: $\eta=0.008, \text{epochs}=60$ with `ReduceLROnPlateau` | Report exact model-specific training hyperparameters. |
| **3** | **M3 Training Loss** | Unweighted sum: $\mathcal{L}_{prelim} + \mathcal{L}_{q1} + \mathcal{L}_{q2}$ | Weighted sum: $\mathcal{L}_{prelim} + 0.6 \mathcal{L}_{q1} + 0.6 \mathcal{L}_{q2}$ with `pos_weight = n_neg / n_pos` | Update Eq. 12 and explain positive class weighting. |
| **4** | **Test Ensemble Thresholds** | Mean of outer fold thresholds ($\bar{\tau} = \frac{1}{5} \sum \tau^{*(k)}$) | Median of outer fold thresholds ($\tau = \text{median}(\tau^{*(k)})$) | Correct text in Section III-F to state median. |
| **5** | **M4 Parametric Matching** | Claims M4 has "matching parametric capacity" | M4 has 1,571 to 1,923 parameters; M3 has 307 to 659 parameters (M4 is $\approx 3\times$ larger) | Remove claim of matching capacity; describe M4 accurately as an over-parameterized classical reference. |
| **6** | **Reported Test Metrics** | Promised Acc, Prec, Rec, F1, ROC-AUC (Sec. IV-F) | Table I omitted Test Accuracy, Precision, Recall | Expand Table I (or Table II) to include the complete test metrics. |
| **7** | **Statistical Testing Interpretation** | "No statistically significant difference detected ($p \ge 0.05$)" | For $N=5$, two-sided Wilcoxon minimum possible $p = 0.0625$ ($2/2^5$). Fixed HQNN beat Adaptive in all 5 folds on Heart and Diabetes. Pooled 15-fold test shows Fixed beats Adaptive with $p = 0.00012$. | Acknowledge that 5-fold test has zero power to detect significance at $\alpha=0.05$ and that Fixed HQNN consistently outperforms Adaptive HQNN in predictive metrics. |
| **8** | **Rotation Gate Definition** | Eq. 6 writes $R(\theta_{l,j})$ with scalar $\theta$; Fig. 1 writes "Rot gates" | Implements PennyLane `qml.Rot` with 3 Euler parameters $(\alpha, \beta, \gamma)$ | Define $R(\alpha, \beta, \gamma) = R_Z(\gamma) R_Y(\beta) R_Z(\alpha)$ explicitly in methodology. |

---
*Report completed under Task A. Awaiting author confirmation before proceeding to Task B or editing paper sources.*
