from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any

from schemas.api_schemas import (
    DatasetMetadata, FeatureInfo, PredictRequest, PredictResponse,
    ExplainRequest, ExplainResponse, ModelPerformanceItem,
    QuantumWorkloadItem, ModelInfoResponse
)
from services.data_service import data_service
from services.model_service import model_service

app = FastAPI(
    title="Adaptive HQNN Medical Disease Prediction API",
    description="Backend API executing the real trained Adaptive Hybrid Quantum-Classical Neural Network across clinical disease models.",
    version="1.0.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Adaptive HQNN API",
        "models_loaded": len(model_service.bundles),
        "supported_models": list(model_service.bundles.keys())
    }

# Model and Dataset Discovery Endpoints
@app.get("/api/models", response_model=List[DatasetMetadata])
@app.get("/api/datasets", response_model=List[DatasetMetadata])
def get_models():
    return data_service.get_all_datasets()

@app.get("/api/models/{model_id}/features")
def get_model_features(model_id: str):
    try:
        ds = data_service.get_dataset(model_id)
        return {
            "model_id": ds.key,
            "display_name": ds.display_name,
            "short_name": ds.short_name,
            "num_features": len(ds.features),
            "features": ds.features,
            "presets": ds.presets
        }
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/api/models/{model_id}", response_model=DatasetMetadata)
def get_model_by_id(model_id: str):
    try:
        return data_service.get_dataset(model_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/api/datasets/{dataset_key}", response_model=DatasetMetadata)
def get_dataset_by_key(dataset_key: str):
    try:
        return data_service.get_dataset(dataset_key)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/api/predict", response_model=PredictResponse)
def run_prediction(req: PredictRequest):
    try:
        return model_service.predict(req)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/api/explain", response_model=ExplainResponse)
def run_explanation(req: ExplainRequest):
    try:
        return model_service.explain(req)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Explanation error: {str(e)}")

@app.get("/api/performance", response_model=List[ModelPerformanceItem])
def get_performance_records():
    return data_service.get_performance_records()

@app.get("/api/workload", response_model=List[QuantumWorkloadItem])
def get_quantum_workload():
    return data_service.get_workload_records()

@app.get("/api/model-info", response_model=ModelInfoResponse)
def get_model_info():
    return data_service.get_model_info()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
