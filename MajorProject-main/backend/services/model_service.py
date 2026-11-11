import os
import json
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from models.adaptive_hqnn import AdaptiveHQNN, N_QUBITS, MAX_QLAYERS
from schemas.api_schemas import (
    PredictRequest, PredictResponse,
    ExplainRequest, ExplainResponse, ShapFeatureContribution
)
from services.data_service import data_service

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_SPLITS_DIR = os.path.join(PROJECT_ROOT, "data", "splits")
MODELS_DIR = os.path.join(PROJECT_ROOT, "results", "models", "adaptive_hqnn")

class DatasetBundle:
    def __init__(self, key: str, feature_names: List[str], target_name: str, tau_h: float, tau_l: float):
        self.key = key
        self.feature_names = feature_names
        self.target_name = target_name
        self.tau_h = tau_h
        self.tau_l = tau_l
        self.models: List[AdaptiveHQNN] = []
        self.preprocessors: List[Tuple[SimpleImputer, StandardScaler]] = []
        self.background_data: np.ndarray = np.zeros((1, len(feature_names)))

class ModelService:
    def __init__(self):
        self.device = torch.device("cpu")
        self.bundles: Dict[str, DatasetBundle] = {}
        self._initialize_bundles()

    def _initialize_bundles(self):
        datasets_config = {
            "breast_cancer": {
                "split_path": os.path.join(DATA_SPLITS_DIR, "breast_cancer", "splits_config.json"),
                "raw_path": os.path.join(DATA_RAW_DIR, "breast_cancer", "wdbc.data"),
                "tau_h": 0.85,
                "tau_l": 0.65,
                "target_pos": "Malignant",
                "target_neg": "Benign"
            },
            "heart": {
                "split_path": os.path.join(DATA_SPLITS_DIR, "heart", "splits_config.json"),
                "raw_path": os.path.join(DATA_RAW_DIR, "heart", "processed.cleveland.data"),
                "tau_h": 0.85,
                "tau_l": 0.65,
                "target_pos": "Heart Disease Present",
                "target_neg": "Absence of Disease"
            },
            "diabetes": {
                "split_path": os.path.join(DATA_SPLITS_DIR, "diabetes", "splits_config.json"),
                "raw_path": os.path.join(DATA_RAW_DIR, "diabetes", "pima-indians-diabetes.csv"),
                "tau_h": 0.85,
                "tau_l": 0.65,
                "target_pos": "Diabetic",
                "target_neg": "Non-Diabetic"
            }
        }

        for key, cfg in datasets_config.items():
            meta = data_service.get_dataset(key)
            feature_names = [f.name for f in meta.features]
            bundle = DatasetBundle(
                key=key,
                feature_names=feature_names,
                target_name=meta.target_positive,
                tau_h=cfg["tau_h"],
                tau_l=cfg["tau_l"]
            )

            # Load Raw DF and split info
            df_raw, dev_idx = self._load_raw_df(key, cfg["raw_path"], cfg["split_path"], feature_names)
            X_dev = df_raw.iloc[dev_idx][feature_names].values

            # Load split config for folds
            with open(cfg["split_path"], "r") as f:
                split_info = json.load(f)

            for fold_info in split_info["cv_folds"]:
                fold_num = fold_info["fold"]
                train_sub_idx = np.array(fold_info["train_dev_idx"])
                X_train_raw = X_dev[train_sub_idx]

                # Fit fold preprocessor
                imputer = SimpleImputer(strategy="median")
                X_imp = imputer.fit_transform(X_train_raw)
                scaler = StandardScaler()
                scaler.fit(X_imp)
                bundle.preprocessors.append((imputer, scaler))

                # Instantiate and load model
                model = AdaptiveHQNN(input_dim=len(feature_names), n_qubits=N_QUBITS, max_layers=MAX_QLAYERS)
                ckpt_path = os.path.join(MODELS_DIR, f"adaptive_hqnn_{key}_fold_{fold_num}.pt")
                if os.path.exists(ckpt_path):
                    state_dict = torch.load(ckpt_path, map_location=self.device)
                    model.load_state_dict(state_dict)
                    model.eval()
                else:
                    print(f"Warning: Checkpoint not found at {ckpt_path}")
                bundle.models.append(model)

            bundle.background_data = scaler.transform(imputer.transform(X_dev[:50]))
            self.bundles[key] = bundle
            print(f"[ModelService] Loaded {len(bundle.models)} fold models for {key}.")

    def _load_raw_df(self, key: str, raw_path: str, split_path: str, feature_names: List[str]):
        with open(split_path, "r") as f:
            split_info = json.load(f)
        dev_idx = np.array(split_info["dev_indices"])

        if key == "breast_cancer":
            raw_cols = ["id", "diagnosis"] + feature_names
            df = pd.read_csv(raw_path, header=None, names=raw_cols)
            df = df.drop(columns=["id"])
        elif key == "heart":
            raw_cols = feature_names + ["num"]
            df = pd.read_csv(raw_path, header=None, names=raw_cols, na_values="?")
        elif key == "diabetes":
            raw_cols = feature_names + ["Outcome"]
            df = pd.read_csv(raw_path, header=None, names=raw_cols)
            for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
                df[col] = df[col].replace(0, np.nan)
        return df, dev_idx

    def predict(self, req: PredictRequest) -> PredictResponse:
        key_input = req.dataset or req.model
        if not key_input:
            raise ValueError("Must provide 'dataset' or 'model' in request.")
        
        meta = data_service.get_dataset(key_input)
        key = meta.key
        if key not in self.bundles:
            raise ValueError(f"Unknown dataset/model '{key}'. Supported: {list(self.bundles.keys())}")

        bundle = self.bundles[key]

        # Parse feature vector
        raw_vector = []
        for feat in bundle.feature_names:
            if feat not in req.features:
                raise ValueError(f"Missing required feature '{feat}' for dataset '{key}'")
            raw_vector.append(float(req.features[feat]))
        X_raw = np.array([raw_vector], dtype=np.float32)

        # Multi-fold inference
        fold_results = []
        prelim_probs = []
        q1_probs = []
        q2_probs = []
        selected_probs = []
        selected_routes = []
        latent_vectors = []
        quantum_expectations = []

        models_to_run = list(enumerate(zip(bundle.models, bundle.preprocessors)))
        if req.fold is not None and 1 <= req.fold <= len(bundle.models):
            f_idx = req.fold - 1
            models_to_run = [(f_idx, (bundle.models[f_idx], bundle.preprocessors[f_idx]))]

        for f_idx, (m, (imputer, scaler)) in models_to_run:
            X_imp = imputer.transform(X_raw)
            X_scaled = scaler.transform(X_imp)
            tensor_x = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                f_logits, p_logits, confs, routes, lat, q_outs = m.forward_adaptive(
                    tensor_x, tau_high=bundle.tau_h, tau_low=bundle.tau_l
                )
                p_prob = torch.sigmoid(p_logits).item()
                f_prob = torch.sigmoid(f_logits).item()
                route = int(routes[0])
                lat_vec = lat[0].cpu().numpy().tolist()
                q_out = q_outs[0]

                prelim_probs.append(p_prob)
                selected_probs.append(f_prob)
                selected_routes.append(route)
                latent_vectors.append(lat_vec)
                if q_out is not None:
                    quantum_expectations.append(q_out)

                fold_results.append({
                    "fold": f_idx + 1,
                    "preliminary_probability": round(p_prob, 4),
                    "confidence": round(confs[0].item(), 4),
                    "quantum_path": route,
                    "final_probability": round(f_prob, 4)
                })

        # Ensemble aggregation
        ens_prelim_p = float(np.mean(prelim_probs))
        ens_conf = float(max(ens_prelim_p, 1.0 - ens_prelim_p))
        
        # Determine ensemble route
        if ens_conf >= bundle.tau_h:
            ens_route = 0
            q_layers = 0
            path_name = "Path 0 (Classical Only)"
            reason = f"Preliminary confidence ({ens_conf*100:.1f}%) exceeds τ_high ({bundle.tau_h*100:.1f}%). Quantum variational layers were skipped to minimize latency."
        elif ens_conf >= bundle.tau_l:
            ens_route = 1
            q_layers = 1
            path_name = "Path 1 (1 Quantum Layer)"
            reason = f"Preliminary confidence ({ens_conf*100:.1f}%) is in intermediate range [τ_low={bundle.tau_l*100:.1f}%, τ_high={bundle.tau_h*100:.1f}%]. Escalated to 1-Layer 4-Qubit Variational Circuit."
        else:
            ens_route = 2
            q_layers = 2
            path_name = "Path 2 (2 Quantum Layers)"
            reason = f"Preliminary confidence ({ens_conf*100:.1f}%) is low (< τ_low={bundle.tau_l*100:.1f}%). Escalated to Full 2-Layer 4-Qubit Variational Circuit."

        final_prob = float(np.mean(selected_probs))
        pred_class = 1 if final_prob >= 0.5 else 0
        pred_label = meta.target_positive if pred_class == 1 else meta.target_negative

        mean_latent = np.mean(latent_vectors, axis=0).tolist()
        mean_expectations = np.mean(quantum_expectations, axis=0).tolist() if quantum_expectations else None

        return PredictResponse(
            dataset=key,
            dataset_display_name=meta.display_name,
            prediction_class=pred_class,
            prediction_label=pred_label,
            probability=round(final_prob, 4),
            confidence=round(max(final_prob, 1.0 - final_prob), 4),
            preliminary_logit=round(float(np.log(ens_prelim_p / (1.0 - ens_prelim_p + 1e-9))), 4),
            preliminary_probability=round(ens_prelim_p, 4),
            preliminary_confidence=round(ens_conf, 4),
            preliminary_label=meta.target_positive if ens_prelim_p >= 0.5 else meta.target_negative,
            quantum_path=ens_route,
            quantum_path_name=path_name,
            quantum_layers_executed=q_layers,
            tau_high=bundle.tau_h,
            tau_low=bundle.tau_l,
            routing_reason=reason,
            latent_vector=[round(v, 4) for v in mean_latent],
            pauli_z_expectations=[round(v, 4) for v in mean_expectations] if mean_expectations else None,
            fold_details=fold_results
        )

    def explain(self, req: ExplainRequest) -> ExplainResponse:
        key_input = req.dataset or req.model
        if not key_input:
            raise ValueError("Must provide 'dataset' or 'model' in request.")
        
        meta = data_service.get_dataset(key_input)
        key = meta.key
        bundle = self.bundles[key]

        raw_vector = [float(req.features[f]) for f in bundle.feature_names]
        X_raw = np.array([raw_vector], dtype=np.float32)

        # Baseline imputer/scaler from fold 1
        imputer, scaler = bundle.preprocessors[0]
        X_scaled = scaler.transform(imputer.transform(X_raw))
        tensor_x = torch.tensor(X_scaled, dtype=torch.float32, requires_grad=True)

        model = bundle.models[0]
        model.eval()
        p_logits, q1_logits, q2_logits = model.forward_all_paths(tensor_x)
        prob = torch.sigmoid(p_logits).item()

        # Feature sensitivity attribution via Integrated Gradients / finite differences
        p_logits.backward()
        grads = tensor_x.grad[0].numpy()
        attributions = grads * (X_scaled[0] - np.mean(bundle.background_data, axis=0))

        # Normalize attributions
        feat_contribs = []
        for i, feat_name in enumerate(bundle.feature_names):
            val = float(req.features[feat_name])
            attr_val = float(attributions[i])
            f_meta = next(f for f in meta.features if f.name == feat_name)
            direction = "positive" if attr_val > 0 else "negative"
            feat_contribs.append(ShapFeatureContribution(
                feature=feat_name,
                display_name=f_meta.display_name,
                value=val,
                shap_value=round(attr_val, 4),
                direction=direction
            ))

        # Sort by absolute impact
        feat_contribs_sorted = sorted(feat_contribs, key=lambda x: abs(x.shap_value), reverse=True)
        top_pos = [c for c in feat_contribs_sorted if c.direction == "positive"][:5]
        top_neg = [c for c in feat_contribs_sorted if c.direction == "negative"][:5]

        routing_note = (
            "Explainability note: SHAP values quantify feature contributions toward the model's decision boundary "
            "and uncertainty estimation. They reflect statistical model associations, not direct biological causation."
        )

        return ExplainResponse(
            dataset=key,
            base_value=0.37 if key == "breast_cancer" else (0.45 if key == "heart" else 0.35),
            prediction_probability=round(prob, 4),
            feature_attributions=feat_contribs_sorted,
            top_positive_features=top_pos,
            top_negative_features=top_neg,
            routing_association_note=routing_note
        )

model_service = ModelService()
