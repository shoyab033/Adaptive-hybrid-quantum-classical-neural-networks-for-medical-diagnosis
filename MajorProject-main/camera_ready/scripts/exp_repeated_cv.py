import os
import sys
import json
import argparse
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
import pennylane as qml

# Force line buffering
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

torch.set_num_threads(min(8, os.cpu_count() or 4))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

REPEATS = 5
DEFAULT_SEEDS = [42, 101, 202, 303, 404]

N_QUBITS = 4
MAX_QLAYERS = 2
q_dev = qml.device('default.qubit', wires=N_QUBITS)

# Quantum circuits
@qml.qnode(q_dev, interface='torch', diff_method='backprop')
def fixed_quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation='Y')
    for l in range(MAX_QLAYERS):
        for i in range(N_QUBITS):
            qml.Rot(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2], wires=i)
        for i in range(N_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % N_QUBITS])
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

@qml.qnode(q_dev, interface='torch', diff_method='backprop')
def dynamic_quantum_circuit_1layer(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation='Y')
    for i in range(N_QUBITS):
        qml.Rot(weights[0, i, 0], weights[0, i, 1], weights[0, i, 2], wires=i)
    for i in range(N_QUBITS):
        qml.CNOT(wires=[i, (i + 1) % N_QUBITS])
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

@qml.qnode(q_dev, interface='torch', diff_method='backprop')
def dynamic_quantum_circuit_2layers(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation='Y')
    for l in range(MAX_QLAYERS):
        for i in range(N_QUBITS):
            qml.Rot(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2], wires=i)
        for i in range(N_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % N_QUBITS])
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


# =============================================================
# Model Architectures
# =============================================================

# M1: Classical MLP
class ClassicalMLP(nn.Module):
    """Architecture: D -> 32 -> LayerNorm -> ReLU -> Dropout(0.2) -> 16 -> ReLU -> 1"""
    def __init__(self, input_dim):
        super(ClassicalMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.norm1 = nn.LayerNorm(32)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(16, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu2(x)
        return self.out(x).squeeze(-1)

# M2: Fixed HQNN
class FixedHQNN(nn.Module):
    """Architecture: D -> 16 -> LayerNorm -> ReLU -> 4 -> Tanh (* pi) -> 2-layer VQC -> Linear(4, 1)"""
    def __init__(self, input_dim):
        super(FixedHQNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Linear(16, N_QUBITS),
            nn.Tanh()
        )
        self.qlayer = qml.qnn.TorchLayer(fixed_quantum_circuit, {"weights": (MAX_QLAYERS, N_QUBITS, 3)})
        self.classifier = nn.Linear(N_QUBITS, 1)
        
    def forward(self, x):
        encoded = self.encoder(x) * np.pi
        q_out = self.qlayer(encoded).float()
        return self.classifier(q_out).squeeze(-1)

# M3: Adaptive HQNN
class AdaptiveHQNN(nn.Module):
    """Architecture: D -> 16 -> prelim head, latent 4, Path 1 (1 Q-layer), Path 2 (2 Q-layers)"""
    def __init__(self, input_dim):
        super(AdaptiveHQNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.LayerNorm(16),
            nn.ReLU()
        )
        self.prelim_head = nn.Linear(16, 1)
        self.to_latent = nn.Sequential(
            nn.Linear(16, N_QUBITS),
            nn.Tanh()
        )
        self.qlayer1 = qml.qnn.TorchLayer(dynamic_quantum_circuit_1layer, {"weights": (1, N_QUBITS, 3)})
        self.qlayer2 = qml.qnn.TorchLayer(dynamic_quantum_circuit_2layers, {"weights": (MAX_QLAYERS, N_QUBITS, 3)})
        self.q1_head = nn.Linear(N_QUBITS, 1)
        self.q2_head = nn.Linear(N_QUBITS, 1)
        
    def forward_all_paths(self, x):
        h = self.encoder(x)
        p_logits = self.prelim_head(h).squeeze(-1)
        latent = self.to_latent(h) * np.pi
        q1_out = self.qlayer1(latent).float()
        q2_out = self.qlayer2(latent).float()
        q1_logits = self.q1_head(q1_out).squeeze(-1)
        q2_logits = self.q2_head(q2_out).squeeze(-1)
        return p_logits, q1_logits, q2_logits
        
    def forward_adaptive(self, x, tau_high=0.85, tau_low=0.65):
        h = self.encoder(x)
        p_logits = self.prelim_head(h).squeeze(-1)
        p_probs = torch.sigmoid(p_logits)
        conf = torch.maximum(p_probs, 1.0 - p_probs)
        latent = self.to_latent(h) * np.pi
        
        N = len(x)
        final_logits = torch.zeros(N, device=x.device)
        routes = np.zeros(N, dtype=int)
        
        for i in range(N):
            conf_i = conf[i].item()
            if conf_i >= tau_high:
                final_logits[i] = p_logits[i]
                routes[i] = 0
            elif conf_i >= tau_low:
                lat_i = latent[i:i+1]
                q1_out = self.qlayer1(lat_i).float()
                final_logits[i] = self.q1_head(q1_out).squeeze(-1)
                routes[i] = 1
            else:
                lat_i = latent[i:i+1]
                q2_out = self.qlayer2(lat_i).float()
                final_logits[i] = self.q2_head(q2_out).squeeze(-1)
                routes[i] = 2
        return final_logits, p_logits, conf, routes

# M4: Adaptive Classical Control
class AdaptiveClassicalControl(nn.Module):
    """Architecture: D -> 16 -> prelim head, c1_refine (16->16->1), c2_refine (16->32->16->1)"""
    def __init__(self, input_dim):
        super(AdaptiveClassicalControl, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.LayerNorm(16),
            nn.ReLU()
        )
        self.prelim_head = nn.Linear(16, 1)
        self.c1_refine = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.c2_refine = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
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

def compute_detailed_metrics(y_true, y_probs):
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    y_pred = (y_probs >= 0.5).astype(int)
    
    acc = accuracy_score(y_true, y_pred) * 100.0
    prec = precision_score(y_true, y_pred, zero_division=0) * 100.0
    rec = recall_score(y_true, y_pred, zero_division=0) * 100.0
    f1 = f1_score(y_true, y_pred, zero_division=0) * 100.0
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = (tn / (tn + fp) * 100.0) if (tn + fp) > 0 else 0.0
    try:
        auc = roc_auc_score(y_true, y_probs)
    except:
        auc = 0.5
    return {
        "accuracy": acc, "precision": prec, "recall": rec, "specificity": spec,
        "f1": f1, "roc_auc": auc, "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
    }

# =============================================================
# Inner CV Threshold Optimization Functions
# =============================================================

def optimize_m4_inner_cv(X_tr_raw, y_tr, input_dim, seed):
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
        m = AdaptiveClassicalControl(input_dim).to(device)
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=device))
        opt = optim.Adam(m.parameters(), lr=0.005, weight_decay=1e-4)
        
        for ep in range(35):
            m.train()
            for bx, by in loader:
                opt.zero_grad()
                pl, c1l, c2l = m.forward_all_paths(bx.to(device))
                loss = crit(pl, by.to(device)) + 0.5 * crit(c1l, by.to(device)) + 0.5 * crit(c2l, by.to(device))
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

def optimize_m3_inner_cv(X_tr_raw, y_tr, input_dim, seed):
    inner_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    oof_p = np.zeros(len(y_tr))
    oof_q1 = np.zeros(len(y_tr))
    oof_q2 = np.zeros(len(y_tr))
    
    for in_fold, (in_tr_idx, in_val_idx) in enumerate(inner_skf.split(X_tr_raw, y_tr)):
        in_prep = FoldPreprocessor()
        X_in_tr = in_prep.fit_transform(X_tr_raw[in_tr_idx])
        X_in_v = in_prep.transform(X_tr_raw[in_val_idx])
        y_in_tr = y_tr[in_tr_idx]
        y_in_v = y_tr[in_val_idx]
        
        pos_w = float(np.sum(y_in_tr==0) / max(1, np.sum(y_in_tr==1)))
        loader = DataLoader(TensorDataset(torch.tensor(X_in_tr, dtype=torch.float32), torch.tensor(y_in_tr, dtype=torch.float32)), batch_size=32, shuffle=True)
        
        torch.manual_seed(seed + in_fold * 10)
        m = AdaptiveHQNN(input_dim).to(device)
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=device))
        opt = optim.Adam(m.parameters(), lr=0.008, weight_decay=1e-4)
        
        for ep in range(25):
            m.train()
            for bx, by in loader:
                opt.zero_grad()
                pl, q1l, q2l = m.forward_all_paths(bx.to(device))
                loss = crit(pl, by.to(device)) + 0.6 * crit(q1l, by.to(device)) + 0.6 * crit(q2l, by.to(device))
                loss.backward()
                opt.step()
                
        m.eval()
        with torch.no_grad():
            pl, q1l, q2l = m.forward_all_paths(torch.tensor(X_in_v, dtype=torch.float32).to(device))
            oof_p[in_val_idx] = torch.sigmoid(pl).cpu().numpy()
            oof_q1[in_val_idx] = torch.sigmoid(q1l).cpu().numpy()
            oof_q2[in_val_idx] = torch.sigmoid(q2l).cpu().numpy()
            
    oof_conf = np.maximum(oof_p, 1.0 - oof_p)
    tau_high_grid = [0.75, 0.80, 0.85, 0.90, 0.95]
    tau_low_grid = [0.55, 0.60, 0.65, 0.70]
    
    best_score = -float('inf')
    best_th, best_tl = 0.85, 0.65
    for th in tau_high_grid:
        for tl in tau_low_grid:
            if th <= tl:
                continue
            sim_probs = np.where(oof_conf >= th, oof_p, np.where(oof_conf >= tl, oof_q1, oof_q2))
            sim_q = np.where(oof_conf >= th, 0, np.where(oof_conf >= tl, 1, 2))
            avg_q = np.mean(sim_q)
            sim_preds = (sim_probs >= 0.5).astype(int)
            f1 = f1_score(y_tr, sim_preds, zero_division=0)
            score = f1 - 0.05 * (avg_q / 2.0)
            if score > best_score:
                best_score = score
                best_th = th
                best_tl = tl
    return best_th, best_tl


# =============================================================
# Main Repeated CV Runner
# =============================================================

def run_repeated_cv(datasets=['breast_cancer', 'heart', 'diabetes'], models=['M1', 'M2', 'M3', 'M4'], seeds=DEFAULT_SEEDS):
    out_raw_csv = "camera_ready/results/repeated_cv_raw_folds.csv"
    
    # Load existing results if resuming
    if os.path.exists(out_raw_csv):
        df_existing = pd.read_csv(out_raw_csv)
        print(f"Found existing raw results: {len(df_existing)} rows. Resuming...")
    else:
        df_existing = pd.DataFrame()
        
    for ds_key in datasets:
        df, feature_names, split_info = load_dataset_data(ds_key)
        ds_name = split_info["display_name"]
        input_dim = len(feature_names)
        
        print(f"\n=======================================================", flush=True)
        print(f"Starting Repeated CV on {ds_name} ({len(seeds)} Repeats x 5 Folds = {len(seeds)*5} Folds)", flush=True)
        print(f"=======================================================", flush=True)
        
        dev_idx = np.array(split_info["dev_indices"])
        X_dev = df.iloc[dev_idx][feature_names].values
        y_dev = df.iloc[dev_idx]['target'].values
        
        for r_idx, seed in enumerate(seeds):
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            
            for f_idx, (tr_idx, val_idx) in enumerate(skf.split(X_dev, y_dev)):
                fold_num = f_idx + 1
                rep_num = r_idx + 1
                
                # Check if all models for this fold already completed
                models_to_run = []
                for m_id in models:
                    if not df_existing.empty:
                        matched = df_existing[
                            (df_existing['dataset'] == ds_key) &
                            (df_existing['model'] == m_id) &
                            (df_existing['repeat'] == rep_num) &
                            (df_existing['fold'] == fold_num)
                        ]
                        if not matched.empty:
                            continue
                    models_to_run.append(m_id)
                    
                if not models_to_run:
                    print(f"[{ds_key}] Repeat {rep_num}, Fold {fold_num} already complete. Skipping.", flush=True)
                    continue
                    
                # Setup fold data
                prep = FoldPreprocessor()
                X_tr_s = prep.fit_transform(X_dev[tr_idx])
                X_v_s = prep.transform(X_dev[val_idx])
                y_tr, y_v = y_dev[tr_idx], y_dev[val_idx]
                
                pos_w = float(np.sum(y_tr==0) / max(1, np.sum(y_tr==1)))
                train_loader = DataLoader(
                    TensorDataset(torch.tensor(X_tr_s, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32)),
                    batch_size=32, shuffle=True
                )
                val_loader = DataLoader(
                    TensorDataset(torch.tensor(X_v_s, dtype=torch.float32), torch.tensor(y_v, dtype=torch.float32)),
                    batch_size=32, shuffle=False
                )
                
                new_records = []
                
                # --- M1: Classical MLP ---
                if 'M1' in models_to_run:
                    torch.manual_seed(seed + fold_num * 10)
                    m1 = ClassicalMLP(input_dim).to(device)
                    crit1 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=device))
                    opt1 = optim.Adam(m1.parameters(), lr=0.003, weight_decay=1e-4)
                    sched1 = optim.lr_scheduler.ReduceLROnPlateau(opt1, mode='min', factor=0.5, patience=6)
                    
                    best_val_loss = float('inf')
                    best_state = None
                    patience_cnt = 0
                    for ep in range(120):
                        m1.train()
                        for bx, by in train_loader:
                            opt1.zero_grad()
                            loss = crit1(m1(bx.to(device)), by.to(device))
                            loss.backward()
                            opt1.step()
                        m1.eval()
                        v_loss = 0.0
                        with torch.no_grad():
                            for bx, by in val_loader:
                                v_loss += crit1(m1(bx.to(device)), by.to(device)).item() * len(by)
                        v_loss /= len(val_loader.dataset)
                        sched1.step(v_loss)
                        if v_loss < best_val_loss - 1e-4:
                            best_val_loss = v_loss
                            best_state = {k: v.cpu().clone() for k, v in m1.state_dict().items()}
                            patience_cnt = 0
                        else:
                            patience_cnt += 1
                            if patience_cnt >= 18:
                                break
                    if best_state is not None:
                        m1.load_state_dict(best_state)
                    m1.eval()
                    with torch.no_grad():
                        probs1 = torch.sigmoid(m1(torch.tensor(X_v_s, dtype=torch.float32).to(device))).cpu().numpy()
                    met1 = compute_detailed_metrics(y_v, probs1)
                    new_records.append({
                        "dataset": ds_key, "dataset_name": ds_name, "model": "M1", "model_name": "Classical MLP (M1)",
                        "repeat": rep_num, "seed": seed, "fold": fold_num,
                        "accuracy": met1["accuracy"], "precision": met1["precision"], "recall": met1["recall"],
                        "specificity": met1["specificity"], "f1": met1["f1"], "roc_auc": met1["roc_auc"],
                        "tp": met1["tp"], "fp": met1["fp"], "tn": met1["tn"], "fn": met1["fn"],
                        "tau_high": np.nan, "tau_low": np.nan,
                        "workload_metric": 0.0, "workload_type": "None"
                    })
                    print(f"  [{ds_key} R{rep_num} F{fold_num}] M1 (Classical MLP): Acc={met1['accuracy']:.2f}%, F1={met1['f1']:.2f}%, AUC={met1['roc_auc']:.4f}", flush=True)

                # --- M2: Fixed HQNN ---
                if 'M2' in models_to_run:
                    torch.manual_seed(seed + fold_num * 10)
                    m2 = FixedHQNN(input_dim).to(device)
                    crit2 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=device))
                    opt2 = optim.Adam(m2.parameters(), lr=0.01, weight_decay=1e-4)
                    sched2 = optim.lr_scheduler.ReduceLROnPlateau(opt2, mode='min', factor=0.5, patience=4)
                    
                    best_val_loss = float('inf')
                    best_state = None
                    patience_cnt = 0
                    for ep in range(60):
                        m2.train()
                        for bx, by in train_loader:
                            opt2.zero_grad()
                            loss = crit2(m2(bx.to(device)), by.to(device))
                            loss.backward()
                            opt2.step()
                        m2.eval()
                        v_loss = 0.0
                        with torch.no_grad():
                            for bx, by in val_loader:
                                v_loss += crit2(m2(bx.to(device)), by.to(device)).item() * len(by)
                        v_loss /= len(val_loader.dataset)
                        sched2.step(v_loss)
                        if v_loss < best_val_loss - 1e-4:
                            best_val_loss = v_loss
                            best_state = {k: v.cpu().clone() for k, v in m2.state_dict().items()}
                            patience_cnt = 0
                        else:
                            patience_cnt += 1
                            if patience_cnt >= 12:
                                break
                    if best_state is not None:
                        m2.load_state_dict(best_state)
                    m2.eval()
                    with torch.no_grad():
                        probs2 = torch.sigmoid(m2(torch.tensor(X_v_s, dtype=torch.float32).to(device))).cpu().numpy()
                    met2 = compute_detailed_metrics(y_v, probs2)
                    new_records.append({
                        "dataset": ds_key, "dataset_name": ds_name, "model": "M2", "model_name": "Fixed HQNN (M2)",
                        "repeat": rep_num, "seed": seed, "fold": fold_num,
                        "accuracy": met2["accuracy"], "precision": met2["precision"], "recall": met2["recall"],
                        "specificity": met2["specificity"], "f1": met2["f1"], "roc_auc": met2["roc_auc"],
                        "tp": met2["tp"], "fp": met2["fp"], "tn": met2["tn"], "fn": met2["fn"],
                        "tau_high": np.nan, "tau_low": np.nan,
                        "workload_metric": 2.0, "workload_type": "Quantum Layers"
                    })
                    print(f"  [{ds_key} R{rep_num} F{fold_num}] M2 (Fixed HQNN): Acc={met2['accuracy']:.2f}%, F1={met2['f1']:.2f}%, AUC={met2['roc_auc']:.4f}", flush=True)

                # --- M4: Adaptive Classical Control ---
                if 'M4' in models_to_run:
                    th4_opt, tl4_opt = optimize_m4_inner_cv(X_dev[tr_idx], y_tr, input_dim, seed=seed+fold_num)
                    torch.manual_seed(seed + fold_num * 10)
                    m4 = AdaptiveClassicalControl(input_dim).to(device)
                    crit4 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=device))
                    opt4 = optim.Adam(m4.parameters(), lr=0.005, weight_decay=1e-4)
                    for ep in range(50):
                        m4.train()
                        for bx, by in train_loader:
                            opt4.zero_grad()
                            pl, c1l, c2l = m4.forward_all_paths(bx.to(device))
                            loss = crit4(pl, by.to(device)) + 0.5 * crit4(c1l, by.to(device)) + 0.5 * crit4(c2l, by.to(device))
                            loss.backward()
                            opt4.step()
                    m4.eval()
                    with torch.no_grad():
                        fl4, _, _, routes4 = m4.forward_adaptive(torch.tensor(X_v_s, dtype=torch.float32).to(device), tau_high=th4_opt, tau_low=tl4_opt)
                        probs4 = torch.sigmoid(fl4).cpu().numpy()
                    met4 = compute_detailed_metrics(y_v, probs4)
                    avg_c4 = float(np.mean(routes4))
                    new_records.append({
                        "dataset": ds_key, "dataset_name": ds_name, "model": "M4", "model_name": "Adaptive Classical Control (M4)",
                        "repeat": rep_num, "seed": seed, "fold": fold_num,
                        "accuracy": met4["accuracy"], "precision": met4["precision"], "recall": met4["recall"],
                        "specificity": met4["specificity"], "f1": met4["f1"], "roc_auc": met4["roc_auc"],
                        "tp": met4["tp"], "fp": met4["fp"], "tn": met4["tn"], "fn": met4["fn"],
                        "tau_high": th4_opt, "tau_low": tl4_opt,
                        "workload_metric": avg_c4, "workload_type": "Refinement Blocks"
                    })
                    print(f"  [{ds_key} R{rep_num} F{fold_num}] M4 (Adaptive Classical): Acc={met4['accuracy']:.2f}%, F1={met4['f1']:.2f}%, AUC={met4['roc_auc']:.4f}, AvgBlocks={avg_c4:.2f}", flush=True)

                # --- M3: Adaptive HQNN ---
                if 'M3' in models_to_run:
                    th3_opt, tl3_opt = optimize_m3_inner_cv(X_dev[tr_idx], y_tr, input_dim, seed=seed+fold_num)
                    torch.manual_seed(seed + fold_num * 10)
                    m3 = AdaptiveHQNN(input_dim).to(device)
                    crit3 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=device))
                    opt3 = optim.Adam(m3.parameters(), lr=0.008, weight_decay=1e-4)
                    sched3 = optim.lr_scheduler.ReduceLROnPlateau(opt3, mode='min', factor=0.5, patience=4)
                    
                    best_val_loss = float('inf')
                    best_state = None
                    patience_cnt = 0
                    for ep in range(60):
                        m3.train()
                        for bx, by in train_loader:
                            opt3.zero_grad()
                            pl, q1l, q2l = m3.forward_all_paths(bx.to(device))
                            loss = crit3(pl, by.to(device)) + 0.6 * crit3(q1l, by.to(device)) + 0.6 * crit3(q2l, by.to(device))
                            loss.backward()
                            opt3.step()
                        m3.eval()
                        v_loss = 0.0
                        with torch.no_grad():
                            for bx, by in val_loader:
                                pl, q1l, q2l = m3.forward_all_paths(bx.to(device))
                                v_loss += (crit3(pl, by.to(device)) + 0.6 * crit3(q1l, by.to(device)) + 0.6 * crit3(q2l, by.to(device))).item() * len(by)
                        v_loss /= len(val_loader.dataset)
                        sched3.step(v_loss)
                        if v_loss < best_val_loss - 1e-4:
                            best_val_loss = v_loss
                            best_state = {k: v.cpu().clone() for k, v in m3.state_dict().items()}
                            patience_cnt = 0
                        else:
                            patience_cnt += 1
                            if patience_cnt >= 10:
                                break
                    if best_state is not None:
                        m3.load_state_dict(best_state)
                    m3.eval()
                    with torch.no_grad():
                        fl3, _, _, routes3 = m3.forward_adaptive(torch.tensor(X_v_s, dtype=torch.float32).to(device), tau_high=th3_opt, tau_low=tl3_opt)
                        probs3 = torch.sigmoid(fl3).cpu().numpy()
                    met3 = compute_detailed_metrics(y_v, probs3)
                    avg_q3 = float(np.mean(routes3))
                    new_records.append({
                        "dataset": ds_key, "dataset_name": ds_name, "model": "M3", "model_name": "Adaptive HQNN (M3)",
                        "repeat": rep_num, "seed": seed, "fold": fold_num,
                        "accuracy": met3["accuracy"], "precision": met3["precision"], "recall": met3["recall"],
                        "specificity": met3["specificity"], "f1": met3["f1"], "roc_auc": met3["roc_auc"],
                        "tp": met3["tp"], "fp": met3["fp"], "tn": met3["tn"], "fn": met3["fn"],
                        "tau_high": th3_opt, "tau_low": tl3_opt,
                        "workload_metric": avg_q3, "workload_type": "Quantum Layers"
                    })
                    print(f"  [{ds_key} R{rep_num} F{fold_num}] M3 (Adaptive HQNN): Acc={met3['accuracy']:.2f}%, F1={met3['f1']:.2f}%, AUC={met3['roc_auc']:.4f}, AvgQ={avg_q3:.2f}", flush=True)

                # Append to existing records and save immediately
                if new_records:
                    df_new = pd.DataFrame(new_records)
                    df_existing = pd.concat([df_existing, df_new], ignore_index=True)
                    df_existing.to_csv(out_raw_csv, index=False)
                    
    # Generate Summary and Statistical Tests
    generate_summary_and_statistical_tests(out_raw_csv)

def generate_summary_and_statistical_tests(raw_csv_path="camera_ready/results/repeated_cv_raw_folds.csv"):
    if not os.path.exists(raw_csv_path):
        print(f"File not found: {raw_csv_path}")
        return
        
    df_raw = pd.read_csv(raw_csv_path)
    print(f"\n=======================================================")
    print(f"Generating Repeated CV Summary and Nadeau-Bengio Statistical Tests")
    print(f"=======================================================")
    
    # 1. Summary Table
    summary_rows = []
    for (ds_key, model_id), grp in df_raw.groupby(['dataset', 'model']):
        ds_name = grp['dataset_name'].iloc[0]
        model_name = grp['model_name'].iloc[0]
        n_evals = len(grp)
        
        def calc_ci(vals):
            m = np.mean(vals)
            s = np.std(vals, ddof=1)
            se = stats.sem(vals)
            h = se * stats.t.ppf((1 + 0.95) / 2., len(vals) - 1)
            return m, s, m - h, m + h
            
        m_acc, s_acc, ci_l_acc, ci_u_acc = calc_ci(grp['accuracy'])
        m_f1, s_f1, ci_l_f1, ci_u_f1 = calc_ci(grp['f1'])
        m_auc, s_auc, ci_l_auc, ci_u_auc = calc_ci(grp['roc_auc'])
        m_rec, s_rec, _, _ = calc_ci(grp['recall'])
        m_spec, s_spec, _, _ = calc_ci(grp['specificity'])
        m_work = np.mean(grp['workload_metric'])
        work_type = grp['workload_type'].iloc[0]
        
        summary_rows.append({
            "Dataset": ds_name,
            "Model": model_name,
            "Model ID": model_id,
            "Total Fold Evals": n_evals,
            "Accuracy (%) (Mean ± SD)": f"{m_acc:.2f} ± {s_acc:.2f}",
            "Accuracy 95% CI": f"[{ci_l_acc:.2f}, {ci_u_acc:.2f}]",
            "F1-Score (%) (Mean ± SD)": f"{m_f1:.2f} ± {s_f1:.2f}",
            "F1 95% CI": f"[{ci_l_f1:.2f}, {ci_u_f1:.2f}]",
            "ROC-AUC (Mean ± SD)": f"{m_auc:.4f} ± {s_auc:.4f}",
            "ROC-AUC 95% CI": f"[{ci_l_auc:.4f}, {ci_u_auc:.4f}]",
            "Sensitivity (%) (Mean ± SD)": f"{m_rec:.2f} ± {s_rec:.2f}",
            "Specificity (%) (Mean ± SD)": f"{m_spec:.2f} ± {s_spec:.2f}",
            "Avg Workload Metric": f"{m_work:.3f} ({work_type})"
        })
        
    df_sum = pd.DataFrame(summary_rows)
    out_sum_csv = "camera_ready/results/repeated_cv_summary.csv"
    df_sum.to_csv(out_sum_csv, index=False)
    print(f"\n[OK] Repeated CV Summary saved to {out_sum_csv}")
    print(df_sum.to_string(index=False))
    
    # 2. Pairwise Statistical Tests: M3 vs M2, M3 vs M4, M3 vs M1 separately per dataset
    test_rows = []
    datasets = df_raw['dataset'].unique()
    comparisons = [('M3', 'M2'), ('M3', 'M4'), ('M3', 'M1')]
    metrics_to_test = ['accuracy', 'f1', 'roc_auc']
    
    for ds in datasets:
        df_ds = df_raw[df_raw['dataset'] == ds]
        ds_name = df_ds['dataset_name'].iloc[0]
        
        for m_test, m_base in comparisons:
            grp_test = df_ds[df_ds['model'] == m_test].sort_values(['repeat', 'fold'])
            grp_base = df_ds[df_ds['model'] == m_base].sort_values(['repeat', 'fold'])
            
            # Match by repeat and fold
            merged = pd.merge(grp_test, grp_base, on=['repeat', 'fold'], suffixes=('_test', '_base'))
            J = len(merged)
            if J < 5:
                print(f"Skipping {m_test} vs {m_base} on {ds}: only {J} paired folds.")
                continue
                
            for met in metrics_to_test:
                d = merged[f'{met}_test'].values - merged[f'{met}_base'].values
                mean_d = float(np.mean(d))
                s2 = float(np.var(d, ddof=1))
                
                # Nadeau-Bengio corrected resampled t-test:
                # sigma^2_NB = (1/J + n_val/n_train) * s^2
                # For 5-fold CV, n_val/n_train = 0.25
                n_val_over_n_train = 0.25
                var_nb = (1.0 / J + n_val_over_n_train) * s2
                se_nb = np.sqrt(max(1e-12, var_nb))
                t_nb = mean_d / se_nb
                df_deg = J - 1
                p_nb = float(2.0 * (1.0 - stats.t.cdf(np.abs(t_nb), df=df_deg)))
                
                # 95% Confidence Interval for mean difference
                t_crit = stats.t.ppf(0.975, df=df_deg)
                ci_lower = mean_d - t_crit * se_nb
                ci_upper = mean_d + t_crit * se_nb
                
                # Uncorrected paired t-test for reference
                se_uncorr = np.sqrt(s2 / J) if s2 > 0 else 1e-12
                t_uncorr = mean_d / se_uncorr
                p_uncorr = float(2.0 * (1.0 - stats.t.cdf(np.abs(t_uncorr), df=df_deg)))
                
                # Wilcoxon signed-rank test
                try:
                    res_w = stats.wilcoxon(d)
                    p_wilcox = float(res_w.pvalue)
                except:
                    p_wilcox = np.nan
                    
                test_rows.append({
                    "Dataset": ds_name,
                    "Comparison": f"{m_test} vs {m_base}",
                    "Metric": met.upper(),
                    "Sample Size (Folds J)": J,
                    "Mean Difference (M3 - Base)": f"{mean_d:+.3f}",
                    "95% CI (Nadeau-Bengio)": f"[{ci_lower:+.3f}, {ci_upper:+.3f}]",
                    "t-stat (Nadeau-Bengio)": f"{t_nb:.3f}",
                    "df": df_deg,
                    "p-value (Nadeau-Bengio)": f"{p_nb:.4f}",
                    "p-value (Uncorrected Paired t)": f"{p_uncorr:.4f}",
                    "p-value (Wilcoxon Signed-Rank)": f"{p_wilcox:.4f}" if not np.isnan(p_wilcox) else "N/A",
                    "Statistically Significant (alpha=0.05, NB)": "Yes" if p_nb < 0.05 else "No",
                    "Notes": "Nadeau-Bengio resampled t-test correction (var = (1/J + 0.25)*s^2)"
                })
                
    df_tests = pd.DataFrame(test_rows)
    out_test_csv = "camera_ready/results/repeated_cv_statistical_tests.csv"
    df_tests.to_csv(out_test_csv, index=False)
    print(f"\n[OK] Repeated CV Statistical Tests saved to {out_test_csv}")
    print(df_tests.to_string(index=False))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='breast_cancer,heart,diabetes')
    parser.add_argument('--models', type=str, default='M1,M2,M3,M4')
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    
    ds_list = [d.strip() for d in args.datasets.split(',')]
    model_list = [m.strip() for m in args.models.split(',')]
    seed_list = DEFAULT_SEEDS[:args.repeats]
    
    run_repeated_cv(datasets=ds_list, models=model_list, seeds=seed_list)
