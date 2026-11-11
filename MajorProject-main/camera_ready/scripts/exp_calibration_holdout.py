import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, brier_score_loss
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pennylane as qml

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_QUBITS = 4
MAX_QLAYERS = 2
q_dev = qml.device('default.qubit', wires=N_QUBITS)

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

@qml.qnode(q_dev, interface='torch', diff_method='backprop')
def fixed_quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation='Y')
    for l in range(MAX_QLAYERS):
        for i in range(N_QUBITS):
            qml.Rot(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2], wires=i)
        for i in range(N_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % N_QUBITS])
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

# Models
class ClassicalMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, dropout_rate=0.2):
        super(ClassicalMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(hidden_dim // 2, 1)
    def forward(self, x):
        x = self.dropout(self.relu1(self.norm1(self.fc1(x))))
        x = self.relu2(self.fc2(x))
        return self.out(x).squeeze(-1)

class FixedHQNN(nn.Module):
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
        enc = self.encoder(x) * np.pi
        q_out = self.qlayer(enc).float()
        return self.classifier(q_out).squeeze(-1)

class AdaptiveHQNN(nn.Module):
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

def compute_ece(y_true, y_probs, n_bins=10):
    """
    Expected Calibration Error (ECE) with equal-width bins in confidence space [0.5, 1.0].
    Confidence C_i = max(p_i, 1 - p_i).
    Accuracy in bin = fraction of correct predictions (y_pred == y_true).
    """
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    y_pred = (y_probs >= 0.5).astype(int)
    conf = np.maximum(y_probs, 1.0 - y_probs)
    
    bin_edges = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    N = len(y_true)
    
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (conf >= low) & (conf <= high) if i == n_bins - 1 else (conf >= low) & (conf < high)
        n_b = np.sum(mask)
        if n_b > 0:
            bin_acc = np.mean(y_pred[mask] == y_true[mask])
            bin_conf = np.mean(conf[mask])
            ece += (n_b / N) * np.abs(bin_acc - bin_conf)
            
    return float(ece)

def bootstrap_ci(y_true, y_probs, n_boot=1000, alpha=0.05):
    """Compute 95% bootstrap confidence intervals for F1 and ROC-AUC."""
    rng = np.random.RandomState(42)
    f1_boots = []
    auc_boots = []
    N = len(y_true)
    
    for _ in range(n_boot):
        idx = rng.randint(0, N, N)
        yb = y_true[idx]
        pb = y_probs[idx]
        pred_b = (pb >= 0.5).astype(int)
        
        # Guard against single-class bootstrap sample
        if len(np.unique(yb)) < 2:
            continue
            
        f1_boots.append(f1_score(yb, pred_b, zero_division=0))
        auc_boots.append(roc_auc_score(yb, pb))
        
    f1_low = float(np.percentile(f1_boots, 100 * (alpha / 2)))
    f1_high = float(np.percentile(f1_boots, 100 * (1 - alpha / 2)))
    auc_low = float(np.percentile(auc_boots, 100 * (alpha / 2)))
    auc_high = float(np.percentile(auc_boots, 100 * (1 - alpha / 2)))
    
    return f1_low, f1_high, auc_low, auc_high

# Load metrics and thresholds
df_m3_metrics = pd.read_csv("results/metrics/m3_adaptive_hqnn_metrics.csv")
df_m4_metrics = pd.read_csv("results/metrics/m4_adaptive_classical_metrics.csv")
df_m4_workload = pd.read_csv("camera_ready/results/m4_workload_metrics.csv")

detailed_test_records = []
calibration_records = []

for ds_key in ['breast_cancer', 'heart', 'diabetes']:
    df, feature_names, split_info = load_dataset_data(ds_key)
    ds_name = split_info["display_name"]
    print(f"\nEvaluating Holdout & Calibration for {ds_name}...")
    
    dev_idx = np.array(split_info["dev_indices"])
    test_idx = np.array(split_info["test_indices"])
    X_dev = df.iloc[dev_idx][feature_names].values
    y_dev = df.iloc[dev_idx]['target'].values
    X_test = df.iloc[test_idx][feature_names].values
    y_test = df.iloc[test_idx]['target'].values
    
    # Preprocessors and model containers
    preprocs = []
    m1_models = []
    m2_models = []
    m3_models = []
    
    for fold_info in split_info["cv_folds"]:
        fn = fold_info["fold"]
        train_sub = np.array(fold_info["train_dev_idx"])
        prep = FoldPreprocessor()
        prep.fit(X_dev[train_sub])
        preprocs.append(prep)
        
        # M1
        m1 = ClassicalMLP(len(feature_names))
        m1.load_state_dict(torch.load(f"results/models/classical_mlp/mlp_{ds_key}_fold_{fn}.pt", map_location=device))
        m1.to(device).eval()
        m1_models.append(m1)
        
        # M2
        m2 = FixedHQNN(len(feature_names))
        m2.load_state_dict(torch.load(f"results/models/fixed_hqnn/fixed_hqnn_{ds_key}_fold_{fn}.pt", map_location=device))
        m2.to(device).eval()
        m2_models.append(m2)
        
        # M3
        m3 = AdaptiveHQNN(len(feature_names))
        m3.load_state_dict(torch.load(f"results/models/adaptive_hqnn/adaptive_hqnn_{ds_key}_fold_{fn}.pt", map_location=device))
        m3.to(device).eval()
        m3_models.append(m3)
        
    # 1. Ensemble Predictions on Holdout Test Set
    # M1 Ensemble
    m1_test_probs = []
    for m, p in zip(m1_models, preprocs):
        with torch.no_grad():
            t_t = torch.tensor(p.transform(X_test), dtype=torch.float32).to(device)
            m1_test_probs.append(torch.sigmoid(m(t_t)).cpu().numpy())
    p_m1_ens = np.mean(m1_test_probs, axis=0)
    
    # M2 Ensemble
    m2_test_probs = []
    for m, p in zip(m2_models, preprocs):
        with torch.no_grad():
            t_t = torch.tensor(p.transform(X_test), dtype=torch.float32).to(device)
            m2_test_probs.append(torch.sigmoid(m(t_t)).cpu().numpy())
    p_m2_ens = np.mean(m2_test_probs, axis=0)
    
    # M3 Ensemble
    m3_prelim_all = []
    m3_q1_all = []
    m3_q2_all = []
    for m, p in zip(m3_models, preprocs):
        with torch.no_grad():
            t_t = torch.tensor(p.transform(X_test), dtype=torch.float32).to(device)
            pl, q1l, q2l = m.forward_all_paths(t_t)
            m3_prelim_all.append(torch.sigmoid(pl).cpu().numpy())
            m3_q1_all.append(torch.sigmoid(q1l).cpu().numpy())
            m3_q2_all.append(torch.sigmoid(q2l).cpu().numpy())
            
    p_prelim_ens = np.mean(m3_prelim_all, axis=0)
    p_q1_ens = np.mean(m3_q1_all, axis=0)
    p_q2_ens = np.mean(m3_q2_all, axis=0)
    c_ens = np.maximum(p_prelim_ens, 1.0 - p_prelim_ens)
    
    th_m3 = float(df_m3_metrics[(df_m3_metrics['dataset']==ds_key) & (df_m3_metrics['fold']=='TEST_ENSEMBLE')]['tau_high'].values[0])
    tl_m3 = float(df_m3_metrics[(df_m3_metrics['dataset']==ds_key) & (df_m3_metrics['fold']=='TEST_ENSEMBLE')]['tau_low'].values[0])
    
    routes_m3 = np.where(c_ens >= th_m3, 0, np.where(c_ens >= tl_m3, 1, 2))
    p_m3_ens = np.where(routes_m3 == 0, p_prelim_ens, np.where(routes_m3 == 1, p_q1_ens, p_q2_ens))
    
    # M4 Ensemble (Load from saved predictions in results or recomputed)
    # Get M4 test row from m4_metrics
    m4_row = df_m4_metrics[(df_m4_metrics['dataset']==ds_key) & (df_m4_metrics['fold']=='TEST_ENSEMBLE')].iloc[0]
    
    # Compute Calibration metrics for Preliminary Head vs M3 Final
    ece_prelim = compute_ece(y_test, p_prelim_ens, n_bins=10)
    brier_prelim = float(brier_score_loss(y_test, p_prelim_ens))
    ece_final = compute_ece(y_test, p_m3_ens, n_bins=10)
    brier_final = float(brier_score_loss(y_test, p_m3_ens))
    
    calibration_records.append({
        "Dataset": ds_name,
        "Preliminary Head ECE": round(ece_prelim, 4),
        "Preliminary Head Brier Score": round(brier_prelim, 4),
        "Adaptive HQNN Final ECE": round(ece_final, 4),
        "Adaptive HQNN Final Brier Score": round(brier_final, 4)
    })
    
    # Compute detailed test suite for M1, M2, M3
    for model_name, p_ens in [
        ('M1: Classical MLP', p_m1_ens),
        ('M2: Fixed HQNN', p_m2_ens),
        ('M3: Adaptive HQNN', p_m3_ens)
    ]:
        pred = (p_ens >= 0.5).astype(int)
        acc = accuracy_score(y_test, pred) * 100
        prec = precision_score(y_test, pred, zero_division=0) * 100
        rec = recall_score(y_test, pred, zero_division=0) * 100 # Sensitivity
        f1 = f1_score(y_test, pred, zero_division=0) * 100
        auc = roc_auc_score(y_test, p_ens)
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        spec = (tn / (tn + fp)) * 100 if (tn + fp) > 0 else 0.0
        
        f1_l, f1_h, auc_l, auc_h = bootstrap_ci(y_test, p_ens, n_boot=1000)
        
        detailed_test_records.append({
            "Dataset": ds_name,
            "Model": model_name,
            "Test N": len(y_test),
            "Accuracy (%)": round(acc, 2),
            "Precision (%)": round(prec, 2),
            "Sensitivity / Recall (%)": round(rec, 2),
            "Specificity (%)": round(spec, 2),
            "F1-Score (%)": round(f1, 2),
            "F1 95% CI": f"[{f1_l*100:.2f}, {f1_h*100:.2f}]",
            "ROC-AUC": round(auc, 4),
            "AUC 95% CI": f"[{auc_l:.4f}, {auc_h:.4f}]",
            "True Pos (TP)": int(tp),
            "False Pos (FP)": int(fp),
            "True Neg (TN)": int(tn),
            "False Neg (FN)": int(fn)
        })
        
    # Append M4
    detailed_test_records.append({
        "Dataset": ds_name,
        "Model": "M4: Adaptive Classical Control",
        "Test N": len(y_test),
        "Accuracy (%)": round(m4_row['accuracy'] * 100, 2),
        "Precision (%)": round(m4_row['precision'] * 100, 2),
        "Sensitivity / Recall (%)": round(m4_row['recall'] * 100, 2),
        "Specificity (%)": round(m4_row['specificity'] * 100, 2),
        "F1-Score (%)": round(m4_row['f1'] * 100, 2),
        "F1 95% CI": "N/A",
        "ROC-AUC": round(m4_row['roc_auc'], 4),
        "AUC 95% CI": "N/A",
        "True Pos (TP)": int(m4_row['tp']),
        "False Pos (FP)": int(m4_row['fp']),
        "True Neg (TN)": int(m4_row['tn']),
        "False Neg (FN)": int(m4_row['fn'])
    })

df_det = pd.DataFrame(detailed_test_records)
out_det_csv = "camera_ready/results/calibration_and_detailed_holdout.csv"
df_det.to_csv(out_det_csv, index=False)

df_cal = pd.DataFrame(calibration_records)
out_cal_csv = "camera_ready/results/confidence_calibration_metrics.csv"
df_cal.to_csv(out_cal_csv, index=False)

print(f"\n[OK] Detailed holdout metrics saved to {out_det_csv}")
print(f"[OK] Confidence calibration metrics saved to {out_cal_csv}")
print("\n--- Detailed Holdout Test Suite ---")
print(df_det[['Dataset', 'Model', 'Accuracy (%)', 'Sensitivity / Recall (%)', 'Specificity (%)', 'F1-Score (%)', 'F1 95% CI', 'ROC-AUC', 'False Neg (FN)']].to_string(index=False))
print("\n--- Calibration Metrics ---")
print(df_cal.to_string(index=False))
