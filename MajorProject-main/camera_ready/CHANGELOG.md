# AIHC 2027 CAMERA-READY REVISION CHANGELOG

**Paper Title:** Adaptive Hybrid Quantum-Classical Neural Network for Early Medical Disease Diagnosis Using Explainable AI  
**Paper ID:** 516 | **Track:** 2 (ML in Clinical Decision Support and Predictive Analytics)  
**Conference:** IEEE Artificial Intelligence in Healthcare Conference (AIHC 2027)  
**Date:** November 1, 2026  
**Revision Branch:** `camera-ready-revision`  
**Camera-Ready PDF:** `camera_ready/CAMERA_READY_REVISED.pdf` (and synchronized `paper/paper.pdf`)  
**Rendered Page Count:** Exactly 5 pages (Strictly within IEEE 6-page maximum limit; eliminates prior Page 6 single-reference overflow)

---

## 1. Executive Summary of Revisions

This camera-ready revision comprehensively addresses all feedback provided across 8 peer reviewers (Reviewers #2, #4, #6, #7, #8, #9, #10, and #11). Revisions include 300 newly completed model runs across 5 repeats of 5-fold cross-validation, a 25-fold ablation suite, an explainability fold-stability audit, state-vector simulation latency benchmarking, rigorous Nadeau–Bengio resampled statistical hypothesis testing with Benjamini–Hochberg False Discovery Rate (FDR) control, and complete correction of all architectural discrepancies.

---

## 2. Detailed Summary of Changes

### A. Architectural and Formal Specification Corrections
1. **Classical MLP (M1) Architecture Specification:**
   - *Previous Text:* Incorrectly stated $D \to 16 \to 8 \to 1$ with Dropout ($p=0.1$).
   - *Revised Text:* Corrected to reflect the exact implementation: $D \to 32 \to 16 \to 1$ with `LayerNorm(32)`, `ReLU()`, and `Dropout(p=0.2)`. Parameter counts updated to 1,601 (WDBC), 1,057 (Heart), and 897 (Diabetes).
2. **Quantum Circuit Formalism and Gate Counts:**
   - *Previous Text:* Variational rotation was loosely written as $R(\theta_{l,j})$ without operator definition; initial state was unspecified; gate counts and depth were absent.
   - *Revised Text:* Formally specified initial ground state $|0000\rangle$, Pauli-$Y$ Angle Embedding ($R_Y(\phi) = \exp(-i\phi\sigma_y/2)$), and 3-parameter single-qubit Euler rotations $R(\alpha, \beta, \gamma) = R_Z(\gamma)R_Y(\beta)R_Z(\alpha)$.
   - Explicitly documented circuit specifications from PennyLane `qml.specs`:
     - Path 1 ($L_Q=1$): Depth 6, 9 operations (4 $R_Y$, 4 `Rot`, 4 circular CNOTs), decomposing into 20 elementary gates with 12 trainable parameters.
     - Path 2 ($L_Q=2$): Depth 11, 17 operations (4 $R_Y$, 8 `Rot`, 8 circular CNOTs), decomposing into 36 elementary gates with 24 trainable parameters.
3. **Composite Training Loss and Imbalance Weighting:**
   - *Previous Text:* Eq. (12) stated an unweighted sum $\mathcal{L}_{prelim} + \mathcal{L}_{q1} + \mathcal{L}_{q2}$.
   - *Revised Text:* Formally stated Eq. (4) with exact path loss coefficients ($1.0, 0.6, 0.6$) and dynamic positive-class weighting $w_{pos} = N_{neg} / N_{pos}$ incorporated into binary cross-entropy to handle clinical label imbalance.
4. **Holdout Test Routing Protocol:**
   - *Previous Text:* Claimed threshold averaging across folds.
   - *Revised Text:* Corrected to document the median outer-fold thresholds $\tau_{high}^{med} = \text{median}_k(\tau_{high}^{*(k)})$ and $\tau_{low}^{med} = \text{median}_k(\tau_{low}^{*(k)})$ applied to the 5-fold ensemble probability.
5. **M4 Parametric Capacity Clarification:**
   - *Previous Text:* Claimed M4 had "matching parametric capacity" to M3.
   - *Revised Text:* Removed false capacity claim. Clarified that full M4 has 1,571 to 1,923 parameters (~3x M3's 307 to 659 parameters). Added a parameter-matched M4 ablation (339 to 691 parameters) for direct capacity-controlled comparison.

---

### B. Empirical Experimental Enhancements (Task B Verification)
1. **Repeated Cross-Validation (Task B1):**
   - Executed **5 repeats of 5-fold stratified cross-validation** across 5 distinct seeds (`[42, 101, 202, 303, 404]`) for M1, M2, M3, and M4 across all three datasets (WDBC, Heart, Diabetes).
   - Generated **300 full model runs** with nested 3-fold inner cross-validation threshold optimization ($[\tau_{high}, \tau_{low}] \in [0.75, 0.95] \times [0.55, 0.70]$) within every outer training fold.
   - Reported mean $\pm$ standard deviation and 95% confidence intervals for Accuracy, F1-Score, and ROC-AUC in Table II.
2. **Nadeau–Bengio Corrected Resampled $t$-Tests (Task B4):**
   - Replaced flawed 5-fold Wilcoxon signed-rank tests with the **Nadeau–Bengio corrected resampled $t$-test** ($J=25, df=24$):
     $$\hat{\sigma}^2_{NB} = \left(\frac{1}{J} + \frac{n_{val}}{n_{train}}\right) s^2 = \left(\frac{1}{25} + 0.25\right) s^2 = 0.29 s^2$$
   - Table V presents pairwise comparisons against M2, M4, and M1 across all datasets.
   - Proved that across all comparisons with Fixed HQNN (M2) and Classical MLP (M1), **no statistically significant difference was detected** ($p_{NB} \ge 0.05$).
3. **Multiple Testing Correction (FDR Adjustment):**
   - Assessed all 27 pairwise hypothesis tests under Benjamini–Hochberg False Discovery Rate (FDR) control at $q=0.05$.
   - The single unadjusted $p < 0.05$ result (Diabetes ROC-AUC M3 vs M4, $+0.0359, t_{NB}=+2.405, p_{NB}=0.0243$) yields $p_{FDR} = 0.6561$.
   - Explicitly qualified this finding as exploratory in the text and refrained from claiming mathematical equivalence.
4. **Comprehensive 25-Fold Ablation Suite (Task B2):**
   - Evaluated three ablation variants across all 25 folds per dataset (Table IV):
     - *Path-0-Only Baseline:* Classical encoder and preliminary head without quantum escalation.
     - *Classical 4-D Bottleneck:* Tanh projection to 4 dimensions coupled to a compact classical block and linear head.
     - *Parameter-Matched M4:* Classical control constrained to approximately match M3 parameter counts (339–691 parameters).
   - Demonstrated that forcing representations through a classical 4-D bottleneck severely degrades performance ($-5.72\%$ F1 on Heart, $-4.85\%$ F1 on Diabetes), whereas Adaptive HQNN preserves diagnostic discrimination.
5. **Kernel SHAP Fold-Stability Audit (Task B3):**
   - Evaluated Kernel SHAP ($N_{bg}=30, N_{eval}=20$) across all 5 outer-fold model checkpoints on holdout instances.
   - Computed pairwise Spearman rank correlations ($\rho$):
     - Diabetes: High stability ($\rho = 0.757 \pm 0.128$, `Glucose` unanimously ranked #1 in 5/5 folds).
     - Heart Disease: Moderate stability ($\rho = 0.391 \pm 0.244$).
     - WDBC: Moderate stability ($\rho = 0.361 \pm 0.088$).
   - Explicitly concluded that XAI feature rankings in hybrid models represent fold-dependent local attributions rather than invariant causal biomarkers.
6. **State-Vector Simulation Latency Benchmarking (Task B5):**
   - Measured 100 timed CPU iterations of `default.qubit` inference.
   - Path 0 latency: ~0.15–0.23 ms; Path 2 latency: ~22.7–24.2 ms ($98\times$ to $155\times$ speedup in CPU simulation).
   - Explicitly disclaimed physical quantum hardware runtimes and cryogenic energy savings.
7. **Probability Calibration Metrics (Task B7):**
   - Computed Expected Calibration Error (ECE) and Brier scores for holdout evaluation.

---

### C. Scientific and Contextual Boundary Discipline
1. **No Claims of Quantum Supremacy or Advantage:**
   - Removed all assertions of quantum superiority. Acknowledged that Classical MLPs (M1) achieve competitive diagnostic performance.
2. **Workload Framing:**
   - Explicitly defined quantum layer metrics as estimated circuit execution proxies under classical simulation.
3. **Honest Framing of Non-Significance:**
   - Adhered strictly to statistical precision: non-significant $p$-values are reported as "no statistically significant difference was detected" rather than claiming mathematical equivalence.

---

### D. Citation and Literature Modernization
1. **Recent 2022–2026 Literature:**
   - Added K. Bharti et al., *Reviews of Modern Physics* 2022 [3].
   - Added A. Abbas et al., *Nature Computational Science* 2021 [12].
   - Added H.-Y. Huang et al., *Science* 2022 [13].
   - Added J. Xin et al., *IEEE TNNLS* 2024 [20].
   - Added C. Nadeau & Y. Bengio, *Machine Learning* 2003 [23].
2. **Strict Sequential Appearance Order:**
   - Reorganized all 23 references into strictly sequential numeric appearance order [1]–[23].

---

### E. Layout and Camera-Ready Artifacts
1. **Page Budget Compliance:**
   - Total pages: exactly 5 pages ($\le 6$ page limit).
   - Cleanly eliminated the awkward 2-line overflow on Page 6 from the original accepted PDF.
   - All 5 tables and all 23 references are rendered legibly within the 5-page layout.
2. **Synchronized Files:**
   - LaTeX source: `paper/paper.tex`
   - Original backup: `camera_ready/backup/paper_original/paper.tex.orig` and `paper.pdf.orig`
   - Compiled camera-ready PDF: `camera_ready/CAMERA_READY_REVISED.pdf` and `paper/paper.pdf`

