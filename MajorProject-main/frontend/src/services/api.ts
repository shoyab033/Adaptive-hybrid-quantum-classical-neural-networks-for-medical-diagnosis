import axios from 'axios';
import {
  DatasetMetadata,
  PredictResponse,
  ExplainResponse,
  ModelPerformanceItem,
  QuantumWorkloadItem,
  ModelInfoResponse
} from '../types';

const API_BASE = (import.meta as any).env?.VITE_API_URL || '/api';

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const getHealth = async () => {
  const res = await api.get('/health');
  return res.data;
};

export const getModels = async (): Promise<DatasetMetadata[]> => {
  const res = await api.get<DatasetMetadata[]>('/models');
  return res.data;
};

export const getDatasets = async (): Promise<DatasetMetadata[]> => {
  return getModels();
};

export const getModelFeatures = async (modelId: string) => {
  const res = await api.get(`/models/${modelId}/features`);
  return res.data;
};

export const getDatasetByKey = async (key: string): Promise<DatasetMetadata> => {
  const res = await api.get<DatasetMetadata>(`/models/${key}`);
  return res.data;
};

export const runPrediction = async (
  model: string,
  features: Record<string, number>,
  fold?: number
): Promise<PredictResponse> => {
  const res = await api.post<PredictResponse>('/predict', {
    model,
    dataset: model,
    features,
    fold: fold || null,
  });
  return res.data;
};

export const runExplanation = async (
  model: string,
  features: Record<string, number>
): Promise<ExplainResponse> => {
  const res = await api.post<ExplainResponse>('/explain', {
    model,
    dataset: model,
    features,
  });
  return res.data;
};

export const getPerformance = async (): Promise<ModelPerformanceItem[]> => {
  const res = await api.get<ModelPerformanceItem[]>('/performance');
  return res.data;
};

export const getQuantumWorkload = async (): Promise<QuantumWorkloadItem[]> => {
  const res = await api.get<QuantumWorkloadItem[]>('/workload');
  return res.data;
};

export const getModelInfo = async (): Promise<ModelInfoResponse> => {
  const res = await api.get<ModelInfoResponse>('/model-info');
  return res.data;
};
