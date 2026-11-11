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

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AdaptiveClassicalControl(nn.Module):
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
        c1_logits = (p_logits + self.c1_refine(h).squeeze(-1))
        c2_logits = (p_logits + self.c2_refine(h).squeeze(-1))
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
                final_logits[i] = (p_logits[i] + self.c1_refine(h[i:i+1]).squeeze(-1))
                routes[i] = 1
            else:
                final_logits[i] = (p_logits[i] + self.c2_refine(h[i:i+1]).squeeze(-1))
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

def train_adaptive_classical_model(model, train_loader, pos_weight, lr=0.005, weight_decay=1e-4, max_epochs=70):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    for epoch in range(max_epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            p_logits, c1_logits, c2_logits = model.forward_all_paths(batch_x)
            
            loss_p = criterion(p_logits, batch_y)
            loss_c1 = criterion(c1_logits, batch_y)
            loss_c2 = criterion(c2_logits, batch_y)
            
            total_loss = loss_p + 0.5 * loss_c1 + 0.5 * loss_c2
            total_loss.backward()
            optimizer.step()
            
    return model

def optimize_thresholds_classical_inner_cv(X_train_raw, y_train, input_dim, random_seed=42):
    inner_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_seed)
    oof_p_probs = np.zeros(len(y_train))
    oof_c1_probs = np.zeros(len(y_train))
    oof_c2_probs = np.zeros(len(y_train))
    
    for in_fold, (in_train_idx, in_val_idx) in enumerate(inner_skf.split(X_train_raw, y_train)):
        in_preproc = FoldPreprocessor()
        X_in_train = in_preproc.fit_transform(X_train_raw[in_train_idx])
        X_in_val = in_preproc.transform(X_train_raw[in_val_idx])
        y_in_train = y_train[in_train_idx]
        y_in_val = y_train[in_val_idx]
        
        n_pos = np.sum(y_in_train == 1)
        n_neg = np.sum(y_in_train == 0)
        pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0
        
        in_train_loader = DataLoader(
            TensorDataset(torch.tensor(X_in_train, dtype=torch.float32), torch.tensor(y_in_train, dtype=torch.float32)),
            batch_size=32, shuffle=True
        )
        
        torch.manual_seed(random_seed + in_fold * 10)
        in_model = AdaptiveClassicalControl(input_dim=input_dim)
        in_model = train_adaptive_classical_model(in_model, in_train_loader, pos_weight=pos_weight, lr=0.005, max_epochs=50)
        
        in_model.eval()
        with torch.no_grad():
            val_t = torch.tensor(X_in_val, dtype=torch.float32).to(device)
            p_logits, c1_logits, c2_logits = in_model.forward_all_paths(val_t)
            oof_p_probs[in_val_idx] = torch.sigmoid(p_logits).cpu().numpy()
            oof_c1_probs[in_val_idx] = torch.sigmoid(c1_logits).cpu().numpy()
            oof_c2_probs[in_val_idx] = torch.sigmoid(c2_logits).cpu().numpy()
            
    oof_conf = np.maximum(oof_p_probs, 1.0 - oof_p_probs)
    
    tau_high_grid = [0.75, 0.80, 0.85, 0.90, 0.95]
    tau_low_grid = [0.55, 0.60, 0.65, 0.70]
    
    best_score = -float('inf')
    best_tau_high = 0.85
    best_tau_low = 0.65
    
    for th in tau_high_grid:
        for tl in tau_low_grid:
            if th <= tl:
                continue
            sim_probs = np.where(oof_conf >= th, oof_p_probs, np.where(oof_conf >= tl, oof_c1_probs, oof_c2_probs))
            sim_c = np.where(oof_conf >= th, 0, np.where(oof_conf >= tl, 1, 2))
            avg_c = np.mean(sim_c)
            
            sim_preds = (sim_probs >= 0.5).astype(int)
            f1 = f1_score(y_train, sim_preds, zero_division=0)
            score = f1 - 0.05 * (avg_c / 2.0)
            
            if score > best_score:
                best_score = score
                best_tau_high = th
                best_tau_low = tl
                
    return best_tau_high, best_tau_low

m4_records = []

for ds_key in ['breast_cancer', 'heart', 'diabetes']:
    df, feature_names, split_info = load_dataset_data(ds_key)
    ds_name = split_info["display_name"]
    print(f"\nProcessing M4 Workload on {ds_name}...")
    
    dev_idx = np.array(split_info["dev_indices"])
    test_idx = np.array(split_info["test_indices"])
    X_dev = df.iloc[dev_idx][feature_names].values
    y_dev = df.iloc[dev_idx]['target'].values
    X_test = df.iloc[test_idx][feature_names].values
    y_test = df.iloc[test_idx]['target'].values
    
    fold_models = []
    fold_preprocs = []
    fold_thresholds = []
    
    fold_r0 = []
    fold_r1 = []
    fold_r2 = []
    fold_avg_blocks = []
    fold_red = []
    
    for fold_info in split_info["cv_folds"]:
        fold_num = fold_info["fold"]
        train_sub = np.array(fold_info["train_dev_idx"])
        val_sub = np.array(fold_info["val_dev_idx"])
        
        X_train_raw = X_dev[train_sub]
        y_train = y_dev[train_sub]
        X_val_raw = X_dev[val_sub]
        y_val = y_dev[val_sub]
        
        th_opt, tl_opt = optimize_thresholds_classical_inner_cv(
            X_train_raw, y_train, input_dim=len(feature_names), random_seed=RANDOM_STATE + fold_num * 100
        )
        fold_thresholds.append((th_opt, tl_opt))
        
        preprocessor = FoldPreprocessor()
        X_train_s = preprocessor.fit_transform(X_train_raw)
        X_val_s = preprocessor.transform(X_val_raw)
        
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0
        
        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train_s, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
            batch_size=32, shuffle=True
        )
        
        torch.manual_seed(RANDOM_STATE + fold_num)
        model = AdaptiveClassicalControl(input_dim=len(feature_names))
        model = train_adaptive_classical_model(model, train_loader, pos_weight=pos_weight)
        
        model.eval()
        with torch.no_grad():
            val_t = torch.tensor(X_val_s, dtype=torch.float32).to(device)
            val_logits, _, _, routes_val = model.forward_adaptive(val_t, tau_high=th_opt, tau_low=tl_opt)
            
        r0 = np.mean(routes_val == 0) * 100
        r1 = np.mean(routes_val == 1) * 100
        r2 = np.mean(routes_val == 2) * 100
        avg_b = np.mean(routes_val)
        red = ((2.0 - avg_b) / 2.0) * 100
        
        fold_r0.append(r0)
        fold_r1.append(r1)
        fold_r2.append(r2)
        fold_avg_blocks.append(avg_b)
        fold_red.append(red)
        
        fold_models.append(model)
        fold_preprocs.append(preprocessor)
        
    # Holdout Test Ensemble
    test_p_all = []
    test_c1_all = []
    test_c2_all = []
    for m, prep in zip(fold_models, fold_preprocs):
        X_test_s = prep.transform(X_test)
        t_t = torch.tensor(X_test_s, dtype=torch.float32).to(device)
        with torch.no_grad():
            p_l, c1_l, c2_l = m.forward_all_paths(t_t)
            test_p_all.append(torch.sigmoid(p_l).cpu().numpy())
            test_c1_all.append(torch.sigmoid(c1_l).cpu().numpy())
            test_c2_all.append(torch.sigmoid(c2_l).cpu().numpy())
            
    ens_p = np.mean(test_p_all, axis=0)
    ens_c1 = np.mean(test_c1_all, axis=0)
    ens_c2 = np.mean(test_c2_all, axis=0)
    ens_conf = np.maximum(ens_p, 1.0 - ens_p)
    
    th_m4_med = float(np.median([th for th, tl in fold_thresholds]))
    tl_m4_med = float(np.median([tl for th, tl in fold_thresholds]))
    
    test_routes_m4 = np.where(ens_conf >= th_m4_med, 0, np.where(ens_conf >= tl_m4_med, 1, 2))
    
    r0_test = np.mean(test_routes_m4 == 0) * 100
    r1_test = np.mean(test_routes_m4 == 1) * 100
    r2_test = np.mean(test_routes_m4 == 2) * 100
    avg_b_test = np.mean(test_routes_m4)
    red_test = ((2.0 - avg_b_test) / 2.0) * 100
    
    m4_records.append({
        "Dataset": ds_name,
        "CV 0-Refine Blocks (%)": f"{np.mean(fold_r0):.1f} ± {np.std(fold_r0):.1f}",
        "CV 1-Refine Block (%)": f"{np.mean(fold_r1):.1f} ± {np.std(fold_r1):.1f}",
        "CV 2-Refine Blocks (%)": f"{np.mean(fold_r2):.1f} ± {np.std(fold_r2):.1f}",
        "CV Avg Blocks/Sample": f"{np.mean(fold_avg_blocks):.2f} ± {np.std(fold_avg_blocks):.2f}",
        "CV Refinement Reduction (%)": f"{np.mean(fold_red):.1f}%",
        "Test 0-Refine (%)": f"{r0_test:.1f}%",
        "Test 1-Refine (%)": f"{r1_test:.1f}%",
        "Test 2-Refine (%)": f"{r2_test:.1f}%",
        "Test Avg Blocks/Sample": f"{avg_b_test:.2f}",
        "Test Refinement Reduction (%)": f"{red_test:.1f}%",
        "Median Thresholds (tau_h, tau_l)": f"({th_m4_med:.2f}, {tl_m4_med:.2f})"
    })

df_m4_out = pd.DataFrame(m4_records)
out_csv = "camera_ready/results/m4_workload_metrics.csv"
df_m4_out.to_csv(out_csv, index=False)
print(f"\n[OK] M4 Workload metrics saved to {out_csv}")
print(df_m4_out.to_string(index=False))
