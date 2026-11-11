import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy import stats

REPEATS = 5
SEEDS = [42, 101, 202, 303, 404]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------------------------------------
# 1. Ablation Model A: Path-0-Only Baseline (Pure Classical)
# -------------------------------------------------------------
class Path0OnlyBaseline(nn.Module):
    """Path-0-only: Shared classical encoder and preliminary head; zero quantum layers."""
    def __init__(self, input_dim):
        super(Path0OnlyBaseline, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.LayerNorm(16),
            nn.ReLU()
        )
        self.head = nn.Linear(16, 1)
        
    def forward(self, x):
        h = self.encoder(x)
        return self.head(h).squeeze(-1)

# -------------------------------------------------------------
# 2. Ablation Model B: Classical 4-D Bottleneck Ablation
# -------------------------------------------------------------
class Classical4dBottleneckAblation(nn.Module):
    """Classical 4-D Bottleneck: Identical 16->4 tanh latent projection, replacing VQC with classical 4-d block."""
    def __init__(self, input_dim):
        super(Classical4dBottleneckAblation, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.LayerNorm(16),
            nn.ReLU()
        )
        self.to_latent = nn.Sequential(
            nn.Linear(16, 4),
            nn.Tanh()
        )
        self.classical_block = nn.Sequential(
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 4)
        )
        self.head = nn.Linear(4, 1)
        
    def forward(self, x):
        h = self.encoder(x)
        lat = self.to_latent(h) * np.pi
        c_out = self.classical_block(lat)
        return self.head(c_out).squeeze(-1)

# -------------------------------------------------------------
# 3. Ablation Model C: Parameter-Matched M4 Adaptive Classical
# -------------------------------------------------------------
class ParameterMatchedM4(nn.Module):
    """Parameter-Matched M4: Capacity reduced to match M3 (approx 339-691 parameters)."""
    def __init__(self, input_dim):
        super(ParameterMatchedM4, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.LayerNorm(16),
            nn.ReLU()
        )
        self.prelim_head = nn.Linear(16, 1)
        self.c1_refine = nn.Sequential(
            nn.Linear(16, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )
        self.c2_refine = nn.Sequential(
            nn.Linear(16, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )
        
    def forward_all_paths(self, x):
        h = self.encoder(x)
        p_logits = self.prelim_head(h).squeeze(-1)
        c1_logits = p_logits + self.c1_refine(h).squeeze(-1)
        c2_logits = p_logits + self.c2_refine(h).squeeze(-1)
        return p_logits, c1_logits, c2_logits
        
    def forward_adaptive(self, x, tau_high=0.85, tau_low=0.65):
        h = self.encoder(x)
        p_logits = self.prelim_head(h).squeeze(-1)
        p_probs = torch.sigmoid(p_logits)
        conf = torch.maximum(p_probs, 1.0 - p_probs)
        
        N = len(x)
        final_logits = torch.zeros(N, device=x.device)
        routes = np.zeros(N, dtype=int)
        
        for i in range(N):
            conf_i = conf[i].item()
            if conf_i >= tau_high:
                final_logits[i] = p_logits[i]
                routes[i] = 0
            elif conf_i >= tau_low:
                final_logits[i] = p_logits[i] + self.c1_refine(h[i:i+1]).squeeze(-1)
                routes[i] = 1
            else:
                final_logits[i] = p_logits[i] + self.c2_refine(h[i:i+1]).squeeze(-1)
                routes[i] = 2
                
        return final_logits, p_logits, conf, routes

class FoldPreprocessor:
    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
    def fit(self, X_train):
        self.imputer.fit(X_train)
        X_imp = self.imputer.transform(X_train)
        self.scaler.fit(X_imp)
        return self
    def transform(self, X):
        X_imp = self.imputer.transform(X)
        return self.scaler.transform(X_imp)
    def fit_transform(self, X_train):
        return self.fit(X_train).transform(X_train)

def load_dataset_data(dataset_key):
    split_path = os.path.join("data", "splits", dataset_key, "splits_config.json")
    with open(split_path, "r") as f:
        split_info = json.load(f)
        
    if dataset_key == "breast_cancer":
        raw_file = os.path.join("data", "raw", "breast_cancer", "wdbc.data")
        feature_names = [
            'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
            'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
            'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
            'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
            'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
            'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
        ]
        raw_cols = ['id', 'diagnosis'] + feature_names
        df_raw = pd.read_csv(raw_file, header=None, names=raw_cols)
        df = df_raw.drop(columns=['id']).copy()
        df['target'] = df['diagnosis'].map({'B': 0, 'M': 1}).astype(int)
        df = df.drop(columns=['diagnosis'])
        
    elif dataset_key == "heart":
        raw_file = os.path.join("data", "raw", "heart", "processed.cleveland.data")
        feature_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        raw_cols = feature_names + ['num']
        df_raw = pd.read_csv(raw_file, header=None, names=raw_cols, na_values='?')
        df = df_raw.copy()
        df['target'] = (df['num'] > 0).astype(int)
        df = df.drop(columns=['num'])
        
    elif dataset_key == "diabetes":
        raw_file = os.path.join("data", "raw", "diabetes", "pima-indians-diabetes.csv")
        feature_names = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        raw_cols = feature_names + ['Outcome']
        df_raw = pd.read_csv(raw_file, header=None, names=raw_cols)
        df = df_raw.copy()
        df['target'] = df['Outcome'].astype(int)
        df = df.drop(columns=['Outcome'])
        for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
            df[col] = df[col].replace(0, np.nan)
            
    return df, feature_names, split_info

def compute_metrics(y_true, y_probs):
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    y_pred = (y_probs >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred) * 100.0
    f1 = f1_score(y_true, y_pred, zero_division=0) * 100.0
    try:
        auc = roc_auc_score(y_true, y_probs)
    except:
        auc = 0.5
    return acc, f1, auc

def optimize_thresholds_pm_m4_inner_cv(X_tr_raw, y_tr, input_dim, seed):
    inner_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    oof_p = np.zeros(len(y_tr))
    oof_c1 = np.zeros(len(y_tr))
    oof_c2 = np.zeros(len(y_tr))
    
    for in_fold, (in_tr_idx, in_val_idx) in enumerate(inner_skf.split(X_tr_raw, y_tr)):
        in_prep = FoldPreprocessor()
        X_in_tr = in_prep.fit_transform(X_tr_raw[in_tr_idx])
        X_in_v = in_prep.transform(X_tr_raw[in_val_idx])
        y_in_tr = y_tr[in_tr_idx]
        y_in_v = y_tr[in_val_idx]
        
        pos_w = float(np.sum(y_in_tr==0) / max(1, np.sum(y_in_tr==1)))
        loader = DataLoader(TensorDataset(torch.tensor(X_in_tr, dtype=torch.float32), torch.tensor(y_in_tr, dtype=torch.float32)), batch_size=32, shuffle=True)
        
        torch.manual_seed(seed + in_fold * 10)
        m = ParameterMatchedM4(input_dim).to(device)
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=device))
        opt = optim.Adam(m.parameters(), lr=0.008, weight_decay=1e-4)
        
        for ep in range(40):
            m.train()
            for bx, by in loader:
                opt.zero_grad()
                pl, c1l, c2l = m.forward_all_paths(bx.to(device))
                loss = crit(pl, by.to(device)) + 0.6 * crit(c1l, by.to(device)) + 0.6 * crit(c2l, by.to(device))
                loss.backward()
                opt.step()
                
        m.eval()
        with torch.no_grad():
            pl, c1l, c2l = m.forward_all_paths(torch.tensor(X_in_v, dtype=torch.float32).to(device))
            oof_p[in_val_idx] = torch.sigmoid(pl).cpu().numpy()
            oof_c1[in_val_idx] = torch.sigmoid(c1l).cpu().numpy()
            oof_c2[in_val_idx] = torch.sigmoid(c2l).cpu().numpy()
            
    oof_conf = np.maximum(oof_p, 1.0 - oof_p)
    tau_high_grid = [0.75, 0.80, 0.85, 0.90, 0.95]
    tau_low_grid = [0.55, 0.60, 0.65, 0.70]
    
    best_score = -float('inf')
    best_th, best_tl = 0.85, 0.65
    for th in tau_high_grid:
        for tl in tau_low_grid:
            if th <= tl:
                continue
            sim_probs = np.where(oof_conf >= th, oof_p, np.where(oof_conf >= tl, oof_c1, oof_c2))
            sim_c = np.where(oof_conf >= th, 0, np.where(oof_conf >= tl, 1, 2))
            avg_c = np.mean(sim_c)
            sim_preds = (sim_probs >= 0.5).astype(int)
            f1 = f1_score(y_tr, sim_preds, zero_division=0)
            score = f1 - 0.05 * (avg_c / 2.0)
            if score > best_score:
                best_score = score
                best_th = th
                best_tl = tl
    return best_th, best_tl

raw_fold_records = []
summary_records = []

for ds_key in ['breast_cancer', 'heart', 'diabetes']:
    df, feature_names, split_info = load_dataset_data(ds_key)
    ds_name = split_info["display_name"]
    input_dim = len(feature_names)
    print(f"\n=======================================================")
    print(f"Running Ablation Suite (5 Repeats x 5-Fold CV) on {ds_name}")
    print(f"=======================================================")
    
    dev_idx = np.array(split_info["dev_indices"])
    test_idx = np.array(split_info["test_indices"])
    X_dev = df.iloc[dev_idx][feature_names].values
    y_dev = df.iloc[dev_idx]['target'].values
    X_test = df.iloc[test_idx][feature_names].values
    y_test = df.iloc[test_idx]['target'].values
    
    # Store metrics across all 25 folds
    m_a_accs, m_a_f1s, m_a_aucs = [], [], []
    m_b_accs, m_b_f1s, m_b_aucs = [], [], []
    m_c_accs, m_c_f1s, m_c_aucs = [], [], []
    m_c_routes_list = []
    
    # Repeat 1 ensemble test predictions (seed 42 models)
    rep1_test_probs_a = []
    rep1_test_probs_b = []
    rep1_test_probs_c = []
    rep1_thresholds_c = []
    
    global_fold_id = 0
    for r_idx, seed in enumerate(SEEDS):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        
        for f_idx, (tr_idx, val_idx) in enumerate(skf.split(X_dev, y_dev)):
            global_fold_id += 1
            fold_num = f_idx + 1
            
            prep = FoldPreprocessor()
            X_tr_s = prep.fit_transform(X_dev[tr_idx])
            X_v_s = prep.transform(X_dev[val_idx])
            X_te_s = prep.transform(X_test)
            y_tr, y_v = y_dev[tr_idx], y_dev[val_idx]
            
            pos_w = float(np.sum(y_tr==0) / max(1, np.sum(y_tr==1)))
            train_loader = DataLoader(
                TensorDataset(torch.tensor(X_tr_s, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32)),
                batch_size=32, shuffle=True
            )
            
            # --- Model A: Path-0-Only Baseline ---
            torch.manual_seed(seed + fold_num * 10)
            mA = Path0OnlyBaseline(input_dim).to(device)
            critA = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=device))
            optA = optim.Adam(mA.parameters(), lr=0.008, weight_decay=1e-4)
            for ep in range(60):
                mA.train()
                for bx, by in train_loader:
                    optA.zero_grad()
                    loss = critA(mA(bx.to(device)), by.to(device))
                    loss.backward()
                    optA.step()
            mA.eval()
            with torch.no_grad():
                v_prob_a = torch.sigmoid(mA(torch.tensor(X_v_s, dtype=torch.float32).to(device))).cpu().numpy()
                te_prob_a = torch.sigmoid(mA(torch.tensor(X_te_s, dtype=torch.float32).to(device))).cpu().numpy()
            acc_a, f1_a, auc_a = compute_metrics(y_v, v_prob_a)
            m_a_accs.append(acc_a); m_a_f1s.append(f1_a); m_a_aucs.append(auc_a)
            if r_idx == 0:
                rep1_test_probs_a.append(te_prob_a)
                
            raw_fold_records.append({
                "dataset": ds_key,
                "dataset_name": ds_name,
                "model": "Path-0-Only Baseline",
                "repeat": r_idx + 1,
                "seed": seed,
                "fold": fold_num,
                "global_fold": global_fold_id,
                "val_accuracy": acc_a,
                "val_f1": f1_a,
                "val_auc": auc_a,
                "workload_metric": 0.0
            })
            
            # --- Model B: Classical 4-D Bottleneck ---
            torch.manual_seed(seed + fold_num * 10)
            mB = Classical4dBottleneckAblation(input_dim).to(device)
            critB = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=device))
            optB = optim.Adam(mB.parameters(), lr=0.008, weight_decay=1e-4)
            for ep in range(60):
                mB.train()
                for bx, by in train_loader:
                    optB.zero_grad()
                    loss = critB(mB(bx.to(device)), by.to(device))
                    loss.backward()
                    optB.step()
            mB.eval()
            with torch.no_grad():
                v_prob_b = torch.sigmoid(mB(torch.tensor(X_v_s, dtype=torch.float32).to(device))).cpu().numpy()
                te_prob_b = torch.sigmoid(mB(torch.tensor(X_te_s, dtype=torch.float32).to(device))).cpu().numpy()
            acc_b, f1_b, auc_b = compute_metrics(y_v, v_prob_b)
            m_b_accs.append(acc_b); m_b_f1s.append(f1_b); m_b_aucs.append(auc_b)
            if r_idx == 0:
                rep1_test_probs_b.append(te_prob_b)
                
            raw_fold_records.append({
                "dataset": ds_key,
                "dataset_name": ds_name,
                "model": "Classical 4-D Bottleneck",
                "repeat": r_idx + 1,
                "seed": seed,
                "fold": fold_num,
                "global_fold": global_fold_id,
                "val_accuracy": acc_b,
                "val_f1": f1_b,
                "val_auc": auc_b,
                "workload_metric": 0.0
            })
            
            # --- Model C: Parameter-Matched M4 (Adaptive Classical) ---
            th_opt, tl_opt = optimize_thresholds_pm_m4_inner_cv(X_dev[tr_idx], y_tr, input_dim, seed=seed+fold_num)
            if r_idx == 0:
                rep1_thresholds_c.append((th_opt, tl_opt))
                
            torch.manual_seed(seed + fold_num * 10)
            mC = ParameterMatchedM4(input_dim).to(device)
            critC = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=device))
            optC = optim.Adam(mC.parameters(), lr=0.008, weight_decay=1e-4)
            for ep in range(60):
                mC.train()
                for bx, by in train_loader:
                    optC.zero_grad()
                    pl, c1l, c2l = mC.forward_all_paths(bx.to(device))
                    loss = critC(pl, by.to(device)) + 0.6 * critC(c1l, by.to(device)) + 0.6 * critC(c2l, by.to(device))
                    loss.backward()
                    optC.step()
            mC.eval()
            with torch.no_grad():
                fl_v, _, _, routes_v = mC.forward_adaptive(torch.tensor(X_v_s, dtype=torch.float32).to(device), tau_high=th_opt, tau_low=tl_opt)
                v_prob_c = torch.sigmoid(fl_v).cpu().numpy()
                fl_te, _, _, _ = mC.forward_adaptive(torch.tensor(X_te_s, dtype=torch.float32).to(device), tau_high=th_opt, tau_low=tl_opt)
                te_prob_c = torch.sigmoid(fl_te).cpu().numpy()
                
            acc_c, f1_c, auc_c = compute_metrics(y_v, v_prob_c)
            avg_routes_c = float(np.mean(routes_v))
            m_c_accs.append(acc_c); m_c_f1s.append(f1_c); m_c_aucs.append(auc_c)
            m_c_routes_list.append(avg_routes_c)
            if r_idx == 0:
                rep1_test_probs_c.append(te_prob_c)
                
            raw_fold_records.append({
                "dataset": ds_key,
                "dataset_name": ds_name,
                "model": "Parameter-Matched M4 (Adaptive Classical)",
                "repeat": r_idx + 1,
                "seed": seed,
                "fold": fold_num,
                "global_fold": global_fold_id,
                "val_accuracy": acc_c,
                "val_f1": f1_c,
                "val_auc": auc_c,
                "workload_metric": avg_routes_c
            })
            
        print(f"  [Repeat {r_idx+1}/5, Seed {seed}] Completed 5 folds.")

    # Calculate Summary Stats across 25 folds
    def get_ci(arr):
        n = len(arr)
        mean = np.mean(arr)
        se = stats.sem(arr)
        h = se * stats.t.ppf((1 + 0.95) / 2., n - 1)
        return mean, np.std(arr), mean - h, mean + h

    # Test holdout ensemble metrics for Repeat 1
    t_ens_a = np.mean(rep1_test_probs_a, axis=0)
    t_acc_a, t_f1_a, t_auc_a = compute_metrics(y_test, t_ens_a)
    
    t_ens_b = np.mean(rep1_test_probs_b, axis=0)
    t_acc_b, t_f1_b, t_auc_b = compute_metrics(y_test, t_ens_b)
    
    # For C, median threshold ensemble
    t_ens_c = np.mean(rep1_test_probs_c, axis=0)
    t_acc_c, t_f1_c, t_auc_c = compute_metrics(y_test, t_ens_c)
    
    # Param counts
    p_a = sum(p.numel() for p in Path0OnlyBaseline(input_dim).parameters())
    p_b = sum(p.numel() for p in Classical4dBottleneckAblation(input_dim).parameters())
    p_c = sum(p.numel() for p in ParameterMatchedM4(input_dim).parameters())
    
    # Summary row for A
    m_acc_a, sd_acc_a, ci_l_acc_a, ci_u_acc_a = get_ci(m_a_accs)
    m_f1_a, sd_f1_a, ci_l_f1_a, ci_u_f1_a = get_ci(m_a_f1s)
    m_auc_a, sd_auc_a, ci_l_auc_a, ci_u_auc_a = get_ci(m_a_aucs)
    summary_records.append({
        "Dataset": ds_name,
        "Ablation Model": "Path-0-Only Baseline (Pure Classical)",
        "Trainable Params": p_a,
        "CV Accuracy (%) (Mean ± SD)": f"{m_acc_a:.2f} ± {sd_acc_a:.2f}",
        "CV Accuracy 95% CI": f"[{ci_l_acc_a:.2f}, {ci_u_acc_a:.2f}]",
        "CV F1-Score (%) (Mean ± SD)": f"{m_f1_a:.2f} ± {sd_f1_a:.2f}",
        "CV F1 95% CI": f"[{ci_l_f1_a:.2f}, {ci_u_f1_a:.2f}]",
        "CV ROC-AUC (Mean ± SD)": f"{m_auc_a:.4f} ± {sd_auc_a:.4f}",
        "CV ROC-AUC 95% CI": f"[{ci_l_auc_a:.4f}, {ci_u_auc_a:.4f}]",
        "Holdout Test Accuracy (%)": f"{t_acc_a:.2f}",
        "Holdout Test F1 (%)": f"{t_f1_a:.2f}",
        "Holdout Test ROC-AUC": f"{t_auc_a:.4f}",
        "Avg Inference Quantum Layers": "0.00",
        "Workload / Complexity Note": "Pure Classical (17-param head)"
    })
    
    # Summary row for B
    m_acc_b, sd_acc_b, ci_l_acc_b, ci_u_acc_b = get_ci(m_b_accs)
    m_f1_b, sd_f1_b, ci_l_f1_b, ci_u_f1_b = get_ci(m_b_f1s)
    m_auc_b, sd_auc_b, ci_l_auc_b, ci_u_auc_b = get_ci(m_b_aucs)
    summary_records.append({
        "Dataset": ds_name,
        "Ablation Model": "Classical 4-D Bottleneck (No Quantum)",
        "Trainable Params": p_b,
        "CV Accuracy (%) (Mean ± SD)": f"{m_acc_b:.2f} ± {sd_acc_b:.2f}",
        "CV Accuracy 95% CI": f"[{ci_l_acc_b:.2f}, {ci_u_acc_b:.2f}]",
        "CV F1-Score (%) (Mean ± SD)": f"{m_f1_b:.2f} ± {sd_f1_b:.2f}",
        "CV F1 95% CI": f"[{ci_l_f1_b:.2f}, {ci_u_f1_b:.2f}]",
        "CV ROC-AUC (Mean ± SD)": f"{m_auc_b:.4f} ± {sd_auc_b:.4f}",
        "CV ROC-AUC 95% CI": f"[{ci_l_auc_b:.4f}, {ci_u_auc_b:.4f}]",
        "Holdout Test Accuracy (%)": f"{t_acc_b:.2f}",
        "Holdout Test F1 (%)": f"{t_f1_b:.2f}",
        "Holdout Test ROC-AUC": f"{t_auc_b:.4f}",
        "Avg Inference Quantum Layers": "0.00",
        "Workload / Complexity Note": "Static Classical 4-D Bottleneck"
    })
    
    # Summary row for C
    m_acc_c, sd_acc_c, ci_l_acc_c, ci_u_acc_c = get_ci(m_c_accs)
    m_f1_c, sd_f1_c, ci_l_f1_c, ci_u_f1_c = get_ci(m_c_f1s)
    m_auc_c, sd_auc_c, ci_l_auc_c, ci_u_auc_c = get_ci(m_c_aucs)
    m_route_c = np.mean(m_c_routes_list)
    summary_records.append({
        "Dataset": ds_name,
        "Ablation Model": "Parameter-Matched M4 (Adaptive Classical)",
        "Trainable Params": p_c,
        "CV Accuracy (%) (Mean ± SD)": f"{m_acc_c:.2f} ± {sd_acc_c:.2f}",
        "CV Accuracy 95% CI": f"[{ci_l_acc_c:.2f}, {ci_u_acc_c:.2f}]",
        "CV F1-Score (%) (Mean ± SD)": f"{m_f1_c:.2f} ± {sd_f1_c:.2f}",
        "CV F1 95% CI": f"[{ci_l_f1_c:.2f}, {ci_u_f1_c:.2f}]",
        "CV ROC-AUC (Mean ± SD)": f"{m_auc_c:.4f} ± {sd_auc_c:.4f}",
        "CV ROC-AUC 95% CI": f"[{ci_l_auc_c:.4f}, {ci_u_auc_c:.4f}]",
        "Holdout Test Accuracy (%)": f"{t_acc_c:.2f}",
        "Holdout Test F1 (%)": f"{t_f1_c:.2f}",
        "Holdout Test ROC-AUC": f"{t_auc_c:.4f}",
        "Avg Inference Quantum Layers": "0.00",
        "Workload / Complexity Note": f"Avg Refine Blocks: {m_route_c:.2f}"
    })

# Save results
df_raw = pd.DataFrame(raw_fold_records)
df_raw.to_csv("camera_ready/results/ablation_raw_folds.csv", index=False)
print(f"\n[OK] Raw ablation fold results saved to camera_ready/results/ablation_raw_folds.csv")

df_summary = pd.DataFrame(summary_records)
df_summary.to_csv("camera_ready/results/ablation_benchmarks.csv", index=False)
print(f"[OK] Summary ablation benchmarks saved to camera_ready/results/ablation_benchmarks.csv")
print(df_summary.to_string(index=False))
