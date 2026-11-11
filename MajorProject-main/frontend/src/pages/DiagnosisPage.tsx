import React, { useState, useEffect } from 'react';
import { 
  getDatasets, 
  runPrediction, 
  runExplanation 
} from '../services/api';
import { 
  DatasetMetadata, 
  PredictResponse, 
  ExplainResponse, 
  PresetSample,
  PredictionHistoryRecord 
} from '../types';
import { FeatureInputForm } from '../components/diagnosis/FeatureInputForm';
import { PredictionCard } from '../components/diagnosis/PredictionCard';
import { AdaptiveRoutingVisualizer } from '../components/diagnosis/AdaptiveRoutingVisualizer';
import { QuantumCircuitViewer } from '../components/diagnosis/QuantumCircuitViewer';
import { 
  Activity, 
  Database, 
  AlertCircle, 
  Sparkles, 
  CheckCircle2, 
  Layers,
  Cpu,
  RotateCcw
} from 'lucide-react';

export const DiagnosisPage: React.FC = () => {
  const [datasets, setDatasets] = useState<DatasetMetadata[]>([]);
  const [selectedDatasetKey, setSelectedDatasetKey] = useState<string>('breast_cancer');
  const [featureValues, setFeatureValues] = useState<Record<string, number>>({});
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [explanation, setExplanation] = useState<ExplainResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [explainLoading, setExplainLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Load datasets on mount
  useEffect(() => {
    const fetchDatasets = async () => {
      try {
        const data = await getDatasets();
        setDatasets(data);
        if (data.length > 0) {
          const initialDs = data[0];
          setSelectedDatasetKey(initialDs.key);
          initFeatureDefaults(initialDs);
        }
      } catch (err: any) {
        setError('Failed to connect to the Adaptive HQNN backend server. Please verify the API is running.');
      }
    };
    fetchDatasets();
  }, []);

  const initFeatureDefaults = (ds: DatasetMetadata) => {
    const defaults: Record<string, number> = {};
    ds.features.forEach((f) => {
      defaults[f.name] = f.default_value;
    });
    setFeatureValues(defaults);
    setPrediction(null);
    setExplanation(null);
    setError(null);
  };

  const currentDataset = datasets.find((d) => d.key === selectedDatasetKey);

  const handleDatasetChange = (key: string) => {
    setSelectedDatasetKey(key);
    const ds = datasets.find((d) => d.key === key);
    if (ds) {
      initFeatureDefaults(ds);
    }
  };

  const handleFeatureChange = (name: string, val: number) => {
    setFeatureValues((prev) => ({ ...prev, [name]: val }));
  };

  const handleSelectPreset = (preset: PresetSample) => {
    setFeatureValues(preset.values);
    setPrediction(null);
    setExplanation(null);
    setError(null);
  };

  const handleReset = () => {
    if (currentDataset) {
      initFeatureDefaults(currentDataset);
    }
  };

  const handleSubmitPrediction = async () => {
    if (!currentDataset) return;
    setLoading(true);
    setError(null);
    try {
      const resp = await runPrediction(currentDataset.key, featureValues);
      setPrediction(resp);
      setExplanation(null);

      // Save to localStorage history
      const historyItem: PredictionHistoryRecord = {
        id: `pred_${Date.now()}`,
        timestamp: new Date().toLocaleString(),
        dataset: currentDataset.key,
        dataset_name: currentDataset.display_name,
        prediction_label: resp.prediction_label,
        probability: resp.probability,
        confidence: resp.confidence,
        quantum_path: resp.quantum_path,
        quantum_layers: resp.quantum_layers_executed,
        features: featureValues,
      };

      const existingHistory = JSON.parse(localStorage.getItem('adaptive_hqnn_history') || '[]');
      localStorage.setItem('adaptive_hqnn_history', JSON.stringify([historyItem, ...existingHistory.slice(0, 49)]));

    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Error executing Adaptive HQNN prediction.');
    } finally {
      setLoading(false);
    }
  };

  const handleFetchExplanation = async () => {
    if (!currentDataset) return;
    setExplainLoading(true);
    try {
      const resp = await runExplanation(currentDataset.key, featureValues);
      setExplanation(resp);
    } catch (err: any) {
      setError('Failed to compute SHAP attributions: ' + (err.message || ''));
    } finally {
      setExplainLoading(false);
    }
  };

  return (
    <div className="space-y-8 py-4">
      {/* Top Title Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-black text-white tracking-tight flex items-center gap-2">
            <Activity className="w-6 h-6 text-brand-400" />
            Interactive Medical Disease Diagnosis
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Real-time inference executing trained 4-Qubit Adaptive HQNN models with uncertainty-guided routing
          </p>
        </div>

        {/* Dataset Switcher Pills */}
        <div className="flex items-center gap-2 bg-slate-900 border border-slate-800 p-1.5 rounded-2xl overflow-x-auto">
          {datasets.map((ds) => (
            <button
              key={ds.key}
              onClick={() => handleDatasetChange(ds.key)}
              className={`px-3 py-1.5 rounded-xl text-xs font-semibold transition-all flex items-center gap-1.5 shrink-0 ${
                selectedDatasetKey === ds.key
                  ? 'bg-brand-600 text-white shadow-md shadow-brand-600/30'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800'
              }`}
            >
              <Database className="w-3.5 h-3.5" />
              {ds.display_name.split(' ')[0]} ({ds.num_features} feats)
            </button>
          ))}
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="bg-rose-500/10 border border-rose-500/30 rounded-xl p-4 flex items-start gap-3 text-rose-300 text-xs">
          <AlertCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
          <div>
            <strong>API Communication Error:</strong> {error}
          </div>
        </div>
      )}

      {/* Main Grid: Form & Results */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* Left Column: Feature Form (7 Cols) */}
        <div className="lg:col-span-7 space-y-6">
          {currentDataset && (
            <FeatureInputForm
              dataset={currentDataset}
              values={featureValues}
              onChange={handleFeatureChange}
              onSelectPreset={handleSelectPreset}
              onResetDefaults={handleReset}
              onSubmit={handleSubmitPrediction}
              loading={loading}
            />
          )}
        </div>

        {/* Right Column: Prediction & Routing Visualizer (5 Cols) */}
        <div className="lg:col-span-5 space-y-6">
          {prediction ? (
            <>
              <PredictionCard
                prediction={prediction}
                onExplainClick={handleFetchExplanation}
              />

              <AdaptiveRoutingVisualizer
                prediction={prediction}
                loading={loading}
              />
            </>
          ) : (
            <div className="glass-panel rounded-2xl p-8 border border-slate-800 text-center space-y-3">
              <div className="w-12 h-12 rounded-full bg-slate-900 border border-slate-700 flex items-center justify-center mx-auto text-slate-500">
                <Layers className="w-6 h-6" />
              </div>
              <h3 className="text-sm font-bold text-white">Awaiting Feature Input</h3>
              <p className="text-xs text-slate-400 leading-relaxed max-w-sm mx-auto">
                Select a clinical preset on the left or customize feature values, then click <strong>"Run Adaptive HQNN Prediction"</strong> to trigger real-time quantum execution.
              </p>
            </div>
          )}
        </div>

      </div>

      {/* Bottom Section: Quantum Circuit & SHAP Attributions (Full Width) */}
      {prediction && (
        <div className="space-y-8 pt-4 border-t border-slate-800">
          <QuantumCircuitViewer
            selectedPath={prediction.quantum_path}
            pauliExpectations={prediction.pauli_z_expectations}
            latentVector={prediction.latent_vector}
          />

          {/* Real-time SHAP Attribution Drawer */}
          {explainLoading && (
            <div className="glass-panel rounded-2xl p-8 text-center text-xs text-brand-300 font-mono">
              Computing Kernel SHAP attributions for active clinical features...
            </div>
          )}

          {explanation && (
            <div className="glass-panel rounded-2xl p-6 border border-slate-800 shadow-xl space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-base font-bold text-white flex items-center gap-2">
                    <Sparkles className="w-5 h-5 text-brand-400" />
                    Patient-Specific SHAP Feature Attribution
                  </h3>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Feature-level impact toward disease risk boundary (Kernel SHAP)
                  </p>
                </div>
                <span className="text-xs font-mono text-slate-400">
                  Base Value: {(explanation.base_value * 100).toFixed(1)}%
                </span>
              </div>

              {/* Top Attributions Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
                {/* Positive (Increasing Risk) */}
                <div className="bg-slate-900/80 p-4 rounded-xl border border-slate-800">
                  <span className="text-xs font-bold text-rose-400 font-mono uppercase tracking-wider block mb-3">
                    ▲ Factors Increasing Predicted Risk
                  </span>
                  <div className="space-y-2">
                    {explanation.top_positive_features.map((f) => (
                      <div key={f.feature} className="flex items-center justify-between text-xs">
                        <span className="text-slate-300 truncate pr-2">{f.display_name} ({f.value})</span>
                        <span className="font-mono font-bold text-rose-400">+{f.shap_value.toFixed(4)}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Negative (Decreasing Risk) */}
                <div className="bg-slate-900/80 p-4 rounded-xl border border-slate-800">
                  <span className="text-xs font-bold text-emerald-400 font-mono uppercase tracking-wider block mb-3">
                    ▼ Factors Decreasing Predicted Risk
                  </span>
                  <div className="space-y-2">
                    {explanation.top_negative_features.map((f) => (
                      <div key={f.feature} className="flex items-center justify-between text-xs">
                        <span className="text-slate-300 truncate pr-2">{f.display_name} ({f.value})</span>
                        <span className="font-mono font-bold text-emerald-400">{f.shap_value.toFixed(4)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <p className="text-[11px] text-slate-400 italic pt-2 border-t border-slate-800">
                {explanation.routing_association_note}
              </p>
            </div>
          )}
        </div>
      )}

    </div>
  );
};
