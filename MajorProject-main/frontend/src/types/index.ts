export interface CategoricalOption {
  value: number;
  label: string;
  description?: string;
}

export interface FeatureInfo {
  name: string;
  display_name: string;
  description: string;
  data_type: string; // 'float' or 'categorical'
  min_value: number;
  max_value: number;
  default_value: number;
  unit?: string;
  step: number;
  category_group?: string;
  options?: CategoricalOption[];
}

export interface PresetSample {
  id: string;
  name: string;
  description: string;
  expected_path: string;
  values: Record<string, number>;
}

export interface DatasetMetadata {
  key: string;
  display_name: string;
  short_name: string;
  domain: string;
  description: string;
  num_samples: number;
  num_features: number;
  target_positive: string;
  target_negative: string;
  class_0_count: number;
  class_1_count: number;
  missing_pct: number;
  tau_high: number;
  tau_low: number;
  features: FeatureInfo[];
  presets: PresetSample[];
}

export interface FoldDetail {
  fold: number;
  preliminary_probability: number;
  confidence: number;
  quantum_path: number;
  final_probability: number;
}

export interface PredictResponse {
  dataset: string;
  dataset_display_name: string;
  prediction_class: number;
  prediction_label: string;
  probability: number;
  confidence: number;
  preliminary_logit: number;
  preliminary_probability: number;
  preliminary_confidence: number;
  preliminary_label: string;
  quantum_path: number; // 0, 1, or 2
  quantum_path_name: string;
  quantum_layers_executed: number; // 0, 1, or 2
  tau_high: number;
  tau_low: number;
  routing_reason: string;
  latent_vector?: number[];
  pauli_z_expectations?: number[];
  fold_details?: FoldDetail[];
}

export interface ShapFeatureContribution {
  feature: string;
  display_name: string;
  value: number;
  shap_value: number;
  direction: 'positive' | 'negative';
}

export interface ExplainResponse {
  dataset: string;
  base_value: number;
  prediction_probability: number;
  feature_attributions: ShapFeatureContribution[];
  top_positive_features: ShapFeatureContribution[];
  top_negative_features: ShapFeatureContribution[];
  routing_association_note: string;
}

export interface ModelPerformanceItem {
  dataset: string;
  model: string;
  model_id: string;
  cv_accuracy: string;
  cv_f1: string;
  cv_roc_auc: string;
  test_f1: string;
  test_roc_auc: string;
}

export interface QuantumWorkloadItem {
  dataset: string;
  classical_only_pct: number;
  one_qlayer_pct: number;
  two_qlayers_pct: number;
  avg_quantum_layers: number;
  workload_reduction_pct: number;
}

export interface ModelInfoResponse {
  project_title: string;
  version: string;
  supported_datasets: string[];
  num_qubits: number;
  max_quantum_layers: number;
  quantum_encoding: string;
  quantum_gates: string[];
  entanglement_topology: string;
  classical_encoder_dims: string;
  readout_method: string;
  citation_bibtex: string;
}

export interface PredictionHistoryRecord {
  id: string;
  timestamp: string;
  dataset: string;
  dataset_name: string;
  prediction_label: string;
  probability: number;
  confidence: number;
  quantum_path: number;
  quantum_layers: number;
  features: Record<string, number>;
}
