import os
import json
import torch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import shap
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

stability_records = []
top_features_per_dataset = {}

for ds_key in ['breast_cancer', 'heart', 'diabetes']:
    df, feature_names, split_info = load_dataset_data(ds_key)
    ds_name = split_info["display_name"]
    print(f"\nAnalyzing SHAP Fold Stability for {ds_name}...")
    
    dev_idx = np.array(split_info["dev_indices"])
    test_idx = np.array(split_info["test_indices"])
    X_dev = df.iloc[dev_idx][feature_names].values
    X_test = df.iloc[test_idx][feature_names].values
    
    fold_shap_rankings = []
    
    for fold_info in split_info["cv_folds"]:
        fold_num = fold_info["fold"]
        train_sub = np.array(fold_info["train_dev_idx"])
        
        preproc = FoldPreprocessor()
        X_train_s = preproc.fit_transform(X_dev[train_sub])
        X_test_s = preproc.transform(X_test)
        
        ckpt_path = f"results/models/adaptive_hqnn/adaptive_hqnn_{ds_key}_fold_{fold_num}.pt"
        model = AdaptiveHQNN(input_dim=len(feature_names))
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)
        model.eval()
        
        th_row = df_m3_metrics[(df_m3_metrics['dataset']==ds_key) & (df_m3_metrics['fold']==str(fold_num))]
        th = float(th_row['tau_high'].values[0])
        tl = float(th_row['tau_low'].values[0])
        
        def model_predict_proba(x_numpy):
            x_t = torch.tensor(x_numpy, dtype=torch.float32).to(device)
            with torch.no_grad():
                final_l, _, _, _ = model.forward_adaptive(x_t, tau_high=th, tau_low=tl)
                p = torch.sigmoid(final_l).cpu().numpy()
            return np.column_stack([1.0 - p, p])
            
        bg_summary = shap.sample(X_train_s, 30, random_state=42)
        explainer = shap.KernelExplainer(model_predict_proba, bg_summary)
        
        eval_count = min(20, len(X_test_s))
        X_eval = X_test_s[:eval_count]
        
        shap_vals = explainer.shap_values(X_eval, nsamples=45)
        if isinstance(shap_vals, list):
            sv = shap_vals[1]
        elif shap_vals.ndim == 3:
            sv = shap_vals[..., 1]
        else:
            sv = shap_vals
            
        mean_abs_shap = np.mean(np.abs(sv), axis=0)
        fold_shap_rankings.append(mean_abs_shap)
        print(f"  Fold {fold_num} complete. Top feature: {feature_names[np.argmax(mean_abs_shap)]}")
        
    # Compute Spearman rank correlations between all pairs of folds (10 pairs)
    spearman_corrs = []
    for i in range(5):
        for j in range(i + 1, 5):
            rho, _ = spearmanr(fold_shap_rankings[i], fold_shap_rankings[j])
            spearman_corrs.append(rho)
            
    mean_rank_shap = np.mean(fold_shap_rankings, axis=0)
    top_indices = np.argsort(mean_rank_shap)[::-1][:5]
    top_feats = [feature_names[idx] for idx in top_indices]
    top_features_per_dataset[ds_name] = top_feats
    
    stability_records.append({
        "Dataset": ds_name,
        "Background Size (N_bg)": 30,
        "Explained Size (N_eval)": 20,
        "Kernel Perturbations (nsamples)": 45,
        "Spearman Rho (Mean ± SD)": f"{np.mean(spearman_corrs):.3f} ± {np.std(spearman_corrs):.3f}",
        "Min Spearman Rho": round(float(np.min(spearman_corrs)), 3),
        "Max Spearman Rho": round(float(np.max(spearman_corrs)), 3),
        "Top 3 Dominant Biomarkers": ", ".join(top_feats[:3])
    })

df_stab = pd.DataFrame(stability_records)
out_csv = "camera_ready/results/shap_fold_stability.csv"
df_stab.to_csv(out_csv, index=False)
print(f"\n[OK] SHAP fold stability results saved to {out_csv}")
print(df_stab.to_string(index=False))
