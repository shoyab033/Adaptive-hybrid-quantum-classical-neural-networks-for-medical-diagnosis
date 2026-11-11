from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class CategoricalOption(BaseModel):
    value: float
    label: str
    description: Optional[str] = None

class FeatureInfo(BaseModel):
    name: str
    display_name: str
    description: str
    data_type: str = "float" # "float" or "categorical"
    min_value: float
    max_value: float
    default_value: float
    unit: Optional[str] = ""
    step: float = 0.1
    category_group: Optional[str] = "Clinical Measurements"
    options: Optional[List[CategoricalOption]] = None

class PresetSample(BaseModel):
    id: str
    name: str
    description: str
    expected_path: str
    values: Dict[str, float]

class DatasetMetadata(BaseModel):
    key: str
    display_name: str
    short_name: str
    domain: str
    description: str
    num_samples: int
    num_features: int
    target_positive: str
    target_negative: str
    class_0_count: int
    class_1_count: int
    missing_pct: float
    tau_high: float
    tau_low: float
    features: List[FeatureInfo]
    presets: List[PresetSample]

class PredictRequest(BaseModel):
    dataset: Optional[str] = Field(None, description="Dataset key: 'breast_cancer', 'heart', or 'diabetes'")
    model: Optional[str] = Field(None, description="Model alias: 'breast_cancer', 'heart', 'heart_disease', or 'diabetes'")
    features: Dict[str, float] = Field(..., description="Feature key-value pairs matching dataset schema")
    fold: Optional[int] = Field(None, description="Optional specific fold (1-5), defaults to ensemble")

class PredictResponse(BaseModel):
    dataset: str
    dataset_display_name: str
    prediction_class: int
    prediction_label: str
    probability: float
    confidence: float
    preliminary_logit: float
    preliminary_probability: float
    preliminary_confidence: float
    preliminary_label: str
    quantum_path: int # 0, 1, or 2
    quantum_path_name: str
    quantum_layers_executed: int # 0, 1, or 2
    tau_high: float
    tau_low: float
    routing_reason: str
    latent_vector: Optional[List[float]] = None
    pauli_z_expectations: Optional[List[float]] = None
    fold_details: Optional[List[Dict[str, Any]]] = None

class ShapFeatureContribution(BaseModel):
    feature: str
    display_name: str
    value: float
    shap_value: float
    direction: str # "positive" (increases disease risk) or "negative" (decreases risk)

class ExplainRequest(BaseModel):
    dataset: Optional[str] = None
    model: Optional[str] = None
    features: Dict[str, float]

class ExplainResponse(BaseModel):
    dataset: str
    base_value: float
    prediction_probability: float
    feature_attributions: List[ShapFeatureContribution]
    top_positive_features: List[ShapFeatureContribution]
    top_negative_features: List[ShapFeatureContribution]
    routing_association_note: str

class ModelPerformanceItem(BaseModel):
    dataset: str
    model: str
    model_id: str
    cv_accuracy: str
    cv_f1: str
    cv_roc_auc: str
    test_f1: str
    test_roc_auc: str

class QuantumWorkloadItem(BaseModel):
    dataset: str
    classical_only_pct: float
    one_qlayer_pct: float
    two_qlayers_pct: float
    avg_quantum_layers: float
    workload_reduction_pct: float

class ModelInfoResponse(BaseModel):
    project_title: str
    version: str
    supported_datasets: List[str]
    num_qubits: int
    max_quantum_layers: int
    quantum_encoding: str
    quantum_gates: List[str]
    entanglement_topology: str
    classical_encoder_dims: str
    readout_method: str
    citation_bibtex: str
