import os
import json
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
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

class AdaptiveHQNN(torch.nn.Module):
    def __init__(self, input_dim, n_qubits=N_QUBITS, max_layers=MAX_QLAYERS):
        super(AdaptiveHQNN, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 16),
            torch.nn.LayerNorm(16),
            torch.nn.ReLU()
        )
        self.prelim_head = torch.nn.Linear(16, 1)
        self.to_latent = torch.nn.Sequential(
            torch.nn.Linear(16, n_qubits),
            torch.nn.Tanh()
        )
        self.qlayer1 = qml.qnn.TorchLayer(dynamic_quantum_circuit_1layer, {"weights": (1, n_qubits, 3)})
        self.qlayer2 = qml.qnn.TorchLayer(dynamic_quantum_circuit_2layers, {"weights": (max_layers, n_qubits, 3)})
        self.q1_head = torch.nn.Linear(n_qubits, 1)
        self.q2_head = torch.nn.Linear(n_qubits, 1)

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

df_m3_metrics = pd.read_csv("results/metrics/m3_adaptive_hqnn_metrics.csv")

per_path_records = []

for ds_key in ['breast_cancer', 'heart', 'diabetes']:
    df, feature_names, split_info = load_dataset_data(ds_key)
    ds_name = split_info["display_name"]
    print(f"\nEvaluating M3 Per-Path Accuracy on {ds_name}...")
    
    dev_idx = np.array(split_info["dev_indices"])
    test_idx = np.array(split_info["test_indices"])
    X_dev = df.iloc[dev_idx][feature_names].values
    y_dev = df.iloc[dev_idx]['target'].values
    X_test = df.iloc[test_idx][feature_names].values
    y_test = df.iloc[test_idx]['target'].values
    
    fold_models = []
    fold_preprocs = []
    
    oof_y_true = []
    oof_prelim_preds = []
    oof_final_preds = []
    oof_routes = []
    
    for fold_info in split_info["cv_folds"]:
        fold_num = fold_info["fold"]
        train_sub = np.array(fold_info["train_dev_idx"])
        val_sub = np.array(fold_info["val_dev_idx"])
        
        preproc = FoldPreprocessor()
        X_train_s = preproc.fit_transform(X_dev[train_sub])
        X_val_s = preproc.transform(X_dev[val_sub])
        y_val = y_dev[val_sub]
        
        ckpt_path = f"results/models/adaptive_hqnn/adaptive_hqnn_{ds_key}_fold_{fold_num}.pt"
        model = AdaptiveHQNN(input_dim=len(feature_names))
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)
        model.eval()
        
        fold_models.append(model)
        fold_preprocs.append(preproc)
        
        th_row = df_m3_metrics[(df_m3_metrics['dataset']==ds_key) & (df_m3_metrics['fold']==str(fold_num))]
        th = float(th_row['tau_high'].values[0])
        tl = float(th_row['tau_low'].values[0])
        
        with torch.no_grad():
            val_t = torch.tensor(X_val_s, dtype=torch.float32).to(device)
            final_l, prelim_l, _, routes = model.forward_adaptive(val_t, tau_high=th, tau_low=tl)
            
            p_probs = torch.sigmoid(prelim_l).cpu().numpy()
            f_probs = torch.sigmoid(final_l).cpu().numpy()
            
            p_pred = (p_probs >= 0.5).astype(int)
            f_pred = (f_probs >= 0.5).astype(int)
            
        oof_y_true.extend(y_val)
        oof_prelim_preds.extend(p_pred)
        oof_final_preds.extend(f_pred)
        oof_routes.extend(routes)
        
    oof_y_true = np.array(oof_y_true)
    oof_prelim_preds = np.array(oof_prelim_preds)
    oof_final_preds = np.array(oof_final_preds)
    oof_routes = np.array(oof_routes)
    
    for r in [0, 1, 2]:
        mask = (oof_routes == r)
        n_samples = int(np.sum(mask))
        if n_samples > 0:
            acc_p = accuracy_score(oof_y_true[mask], oof_prelim_preds[mask]) * 100
            acc_f = accuracy_score(oof_y_true[mask], oof_final_preds[mask]) * 100
            delta = acc_f - acc_p
        else:
            acc_p, acc_f, delta = np.nan, np.nan, np.nan
            
        per_path_records.append({
            "Dataset": ds_name,
            "Evaluation": "5-Fold CV (OOF)",
            "Path": f"Path {r} ({r} Q-Layers)",
            "Sample Count": n_samples,
            "Sample Pct (%)": round((n_samples / len(oof_routes)) * 100, 2),
            "Prelim Accuracy (%)": round(acc_p, 2) if not np.isnan(acc_p) else "N/A",
            "Final Accuracy (%)": round(acc_f, 2) if not np.isnan(acc_f) else "N/A",
            "Accuracy Delta (%)": round(delta, 2) if not np.isnan(delta) else "N/A"
        })
        
    # Holdout Test Ensemble
    test_prelim_all = []
    test_q1_all = []
    test_q2_all = []
    for m, prep in zip(fold_models, fold_preprocs):
        X_test_s = prep.transform(X_test)
        t_t = torch.tensor(X_test_s, dtype=torch.float32).to(device)
        with torch.no_grad():
            p_l, q1_l, q2_l = m.forward_all_paths(t_t)
            test_prelim_all.append(torch.sigmoid(p_l).cpu().numpy())
            test_q1_all.append(torch.sigmoid(q1_l).cpu().numpy())
            test_q2_all.append(torch.sigmoid(q2_l).cpu().numpy())
            
    ens_prelim = np.mean(test_prelim_all, axis=0)
    ens_q1 = np.mean(test_q1_all, axis=0)
    ens_q2 = np.mean(test_q2_all, axis=0)
    ens_conf = np.maximum(ens_prelim, 1.0 - ens_prelim)
    
    th_test = float(df_m3_metrics[(df_m3_metrics['dataset']==ds_key) & (df_m3_metrics['fold']=='TEST_ENSEMBLE')]['tau_high'].values[0])
    tl_test = float(df_m3_metrics[(df_m3_metrics['dataset']==ds_key) & (df_m3_metrics['fold']=='TEST_ENSEMBLE')]['tau_low'].values[0])
    
    test_routes = np.where(ens_conf >= th_test, 0, np.where(ens_conf >= tl_test, 1, 2))
    test_final_probs = np.where(test_routes == 0, ens_prelim, np.where(test_routes == 1, ens_q1, ens_q2))
    
    test_p_pred = (ens_prelim >= 0.5).astype(int)
    test_f_pred = (test_final_probs >= 0.5).astype(int)
    
    for r in [0, 1, 2]:
        mask = (test_routes == r)
        n_samples = int(np.sum(mask))
        if n_samples > 0:
            acc_p = accuracy_score(y_test[mask], test_p_pred[mask]) * 100
            acc_f = accuracy_score(y_test[mask], test_f_pred[mask]) * 100
            delta = acc_f - acc_p
        else:
            acc_p, acc_f, delta = np.nan, np.nan, np.nan
            
        per_path_records.append({
            "Dataset": ds_name,
            "Evaluation": "Holdout Test (Ensemble)",
            "Path": f"Path {r} ({r} Q-Layers)",
            "Sample Count": n_samples,
            "Sample Pct (%)": round((n_samples / len(test_routes)) * 100, 2),
            "Prelim Accuracy (%)": round(acc_p, 2) if not np.isnan(acc_p) else "N/A",
            "Final Accuracy (%)": round(acc_f, 2) if not np.isnan(acc_f) else "N/A",
            "Accuracy Delta (%)": round(delta, 2) if not np.isnan(delta) else "N/A"
        })

df_per_path = pd.DataFrame(per_path_records)
out_csv = "camera_ready/results/m3_per_path_accuracy.csv"
df_per_path.to_csv(out_csv, index=False)
print(f"\n[OK] M3 per-path accuracy successfully saved to {out_csv}")
print(df_per_path.to_string(index=False))
