import React, { useState, useEffect } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { 
  getModels, 
  getModelFeatures, 
  runPrediction, 
  runExplanation 
} from '../services/api';
import { 
  DatasetMetadata, 
  FeatureInfo, 
  PresetSample, 
  PredictResponse, 
  ExplainResponse,
  PredictionHistoryRecord 
} from '../types';
import { 
  Activity, 
  Stethoscope, 
  Heart, 
  Droplet, 
  ShieldCheck, 
  CheckCircle2, 
  AlertCircle, 
  RotateCcw, 
  Sparkles, 
  ArrowRight, 
  ArrowLeft, 
  Cpu, 
  Zap, 
  Info, 
  Sliders, 
  Layers,
  History
} from 'lucide-react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  Tooltip, 
  ResponsiveContainer, 
  CartesianGrid, 
  Cell 
} from 'recharts';

export const PredictionPage: React.FC = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();

  // State
  const [models, setModels] = useState<DatasetMetadata[]>([]);
  const [selectedModelKey, setSelectedModelKey] = useState<string>('diabetes');
  const [currentModel, setCurrentModel] = useState<DatasetMetadata | null>(null);
  const [featureValues, setFeatureValues] = useState<Record<string, number>>({});
  const [validationErrors, setValidationErrors] = useState<Record<string, string>>({});
  const [activePresetId, setActivePresetId] = useState<string | null>(null);

  // Execution states
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [explanation, setExplanation] = useState<ExplainResponse | null>(null);
  const [explainLoading, setExplainLoading] = useState<boolean>(false);

  // Load models on mount
  useEffect(() => {
    const fetchModelList = async () => {
      try {
        const data = await getModels();
        setModels(data);

        // Check query param for initial model
        const modelParam = searchParams.get('model');
        const initialKey = modelParam && data.some(m => m.key === modelParam)
          ? modelParam
          : (data.length > 0 ? data[0].key : 'diabetes');

        setSelectedModelKey(initialKey);
      } catch (err: any) {
        setError('Failed to connect to backend server. Please ensure the FastAPI server is running.');
      }
    };
    fetchModelList();
  }, []);

  // When selected model changes, fetch schema and initialize form
  useEffect(() => {
    if (!selectedModelKey) return;
    const fetchSchema = async () => {
      try {
        const schemaData = await getModelFeatures(selectedModelKey);
        const fullModelMeta = models.find(m => m.key === selectedModelKey) || schemaData;
        setCurrentModel(fullModelMeta);

        // Initialize default feature values
        const defaults: Record<string, number> = {};
        schemaData.features.forEach((f: FeatureInfo) => {
          defaults[f.name] = f.default_value;
        });
        setFeatureValues(defaults);
        setValidationErrors({});
        setActivePresetId(null);
        setPrediction(null);
        setExplanation(null);
        setError(null);
      } catch (err: any) {
        console.error('Error fetching model features', err);
      }
    };
    fetchSchema();
  }, [selectedModelKey, models]);

  // Handle disease card selection
  const handleSelectModel = (key: string) => {
    setSelectedModelKey(key);
    setSearchParams({ model: key });
  };

  // Handle input change
  const handleInputChange = (featureName: string, value: number) => {
    setFeatureValues(prev => ({ ...prev, [featureName]: value }));
    setActivePresetId(null);
    if (validationErrors[featureName]) {
      setValidationErrors(prev => {
        const copy = { ...prev };
        delete copy[featureName];
        return copy;
      });
    }
  };

  // Handle Preset Load
  const handleLoadPreset = (preset: PresetSample) => {
    setActivePresetId(preset.id);
    setFeatureValues(preset.values);
    setValidationErrors({});
    setPrediction(null);
    setExplanation(null);
    setError(null);
  };

  // Reset defaults
  const handleResetDefaults = () => {
    if (!currentModel) return;
    const defaults: Record<string, number> = {};
    currentModel.features.forEach(f => {
      defaults[f.name] = f.default_value;
    });
    setFeatureValues(defaults);
    setValidationErrors({});
    setActivePresetId(null);
    setPrediction(null);
    setExplanation(null);
    setError(null);
  };

  // Validate form
  const validateForm = (): boolean => {
    if (!currentModel) return false;
    const errors: Record<string, string> = {};

    currentModel.features.forEach(f => {
      const val = featureValues[f.name];
      if (val === undefined || isNaN(val)) {
        errors[f.name] = `${f.display_name} is required.`;
      } else if (val < f.min_value || val > f.max_value) {
        errors[f.name] = `Value must be between ${f.min_value} and ${f.max_value} ${f.unit || ''}.`;
      }
    });

    setValidationErrors(errors);
    return Object.keys(errors).length === 0;
  };

  // Submit Prediction
  const handleSubmitPrediction = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!currentModel) return;

    if (!validateForm()) {
      setError('Please review the clinical input parameters highlighted in red.');
      return;
    }

    setLoading(true);
    setError(null);
    setPrediction(null);
    setExplanation(null);

    try {
      // 1. Run actual prediction through backend
      const predResp = await runPrediction(currentModel.key, featureValues);
      setPrediction(predResp);

      // 2. Automatically fetch real Kernel SHAP explanation for the prediction
      try {
        setExplainLoading(true);
        const expResp = await runExplanation(currentModel.key, featureValues);
        setExplanation(expResp);
      } catch (expErr) {
        console.warn('Could not auto-fetch explanation', expErr);
      } finally {
        setExplainLoading(false);
      }

      // 3. Save to local prediction history
      const historyItem: PredictionHistoryRecord = {
        id: `pred_${Date.now()}`,
        timestamp: new Date().toLocaleString(),
        dataset: currentModel.key,
        dataset_name: currentModel.display_name,
        prediction_label: predResp.prediction_label,
        probability: predResp.probability,
        confidence: predResp.confidence,
        quantum_path: predResp.quantum_path,
        quantum_layers: predResp.quantum_layers_executed,
        features: featureValues,
      };
      const existing = JSON.parse(localStorage.getItem('adaptive_hqnn_history') || '[]');
      localStorage.setItem('adaptive_hqnn_history', JSON.stringify([historyItem, ...existing.slice(0, 49)]));

    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Error executing Adaptive HQNN prediction.');
    } finally {
      setLoading(false);
    }
  };

  // Group features by category_group
  const groupedFeatures: Record<string, FeatureInfo[]> = {};
  if (currentModel) {
    currentModel.features.forEach(f => {
      const group = f.category_group || 'Clinical Parameters';
      if (!groupedFeatures[group]) {
        groupedFeatures[group] = [];
      }
      groupedFeatures[group].push(f);
    });
  }

  const getDiseaseIcon = (key: string) => {
    if (key.includes('heart')) return Heart;
    if (key.includes('diabetes')) return Droplet;
    return ShieldCheck;
  };

  return (
    <div className="space-y-10 py-4 max-w-5xl mx-auto">
      
      {/* Top Header */}
      <div className="text-center space-y-2 pb-2">
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-xs font-mono text-cyan-300">
          <Stethoscope className="w-3.5 h-3.5 text-cyan-400" />
          Clinical Inference Engine
        </div>
        <h1 className="text-3xl sm:text-4xl font-black text-white tracking-tight">
          Medical Disease Prediction
        </h1>
        <p className="text-xs text-slate-400 max-w-xl mx-auto">
          Enter patient measurements to execute the real Adaptive HQNN model with dynamic quantum layer routing and Kernel SHAP explainability.
        </p>
      </div>

      {/* Step 1: Select Prediction Type */}
      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-bold text-white uppercase tracking-wider font-mono flex items-center gap-2">
            <span className="w-5 h-5 rounded-full bg-cyan-500/20 text-cyan-400 flex items-center justify-center text-xs font-bold">1</span>
            Select Prediction Type
          </h2>
          <span className="text-xs text-slate-400 font-mono">
            Active: <strong className="text-white">{currentModel?.short_name || 'Select a Disease'}</strong>
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {models.map(m => {
            const Icon = getDiseaseIcon(m.key);
            const isSelected = selectedModelKey === m.key;

            return (
              <button
                key={m.key}
                type="button"
                onClick={() => handleSelectModel(m.key)}
                className={`p-4 rounded-2xl text-left border transition-all duration-200 relative ${
                  isSelected
                    ? 'bg-gradient-to-b from-cyan-950/80 to-blue-950/80 border-cyan-400 shadow-lg shadow-cyan-500/10 ring-2 ring-cyan-500/30'
                    : 'glass-panel border-slate-800 hover:border-slate-700 opacity-80 hover:opacity-100'
                }`}
              >
                {isSelected && (
                  <span className="absolute top-3 right-3 flex h-2.5 w-2.5">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-cyan-500"></span>
                  </span>
                )}
                <div className="flex items-center gap-3">
                  <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${
                    isSelected ? 'bg-cyan-500 text-slate-950' : 'bg-slate-900 text-slate-400'
                  }`}>
                    <Icon className="w-5 h-5" />
                  </div>
                  <div>
                    <h3 className="text-sm font-bold text-white">{m.short_name || m.display_name}</h3>
                    <p className="text-[11px] text-slate-400 font-mono mt-0.5">{m.num_features} features</p>
                  </div>
                </div>
              </button>
            );
          })}
        </div>
      </section>

      {/* Step 2: Patient Clinical Information Form */}
      {currentModel && !prediction && (
        <section className="glass-panel rounded-3xl p-6 sm:p-8 border border-slate-800 shadow-2xl space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
            <div>
              <h2 className="text-base font-bold text-white flex items-center gap-2">
                <span className="w-5 h-5 rounded-full bg-cyan-500/20 text-cyan-400 flex items-center justify-center text-xs font-bold font-mono">2</span>
                Patient Clinical Information ({currentModel.short_name})
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Fill in the patient's measurements or click an instant calibrated preset below
              </p>
            </div>

            <button
              type="button"
              onClick={handleResetDefaults}
              className="self-start sm:self-auto px-3.5 py-1.5 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-700 text-xs text-slate-300 font-medium transition-colors flex items-center gap-1.5"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Reset Defaults
            </button>
          </div>

          {/* Quick-Load Sample Presets */}
          <div className="space-y-2">
            <span className="text-[11px] font-mono text-cyan-300 uppercase tracking-wider font-semibold block">
              ⭐ Instant Clinical Sample Presets (For Live Testing)
            </span>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
              {currentModel.presets.map(preset => (
                <button
                  key={preset.id}
                  type="button"
                  onClick={() => handleLoadPreset(preset)}
                  className={`p-3 rounded-xl text-left border transition-all text-xs ${
                    activePresetId === preset.id
                      ? 'bg-cyan-500/20 border-cyan-400 text-white shadow-md shadow-cyan-500/10'
                      : 'bg-slate-900/90 border-slate-800 text-slate-300 hover:bg-slate-850 hover:border-slate-700'
                  }`}
                >
                  <div className="font-bold text-white flex items-center justify-between">
                    <span>{preset.name}</span>
                    {activePresetId === preset.id && <CheckCircle2 className="w-4 h-4 text-cyan-400" />}
                  </div>
                  <p className="text-[11px] text-slate-400 mt-1 line-clamp-2 leading-relaxed">
                    {preset.description}
                  </p>
                  <span className="inline-block mt-2 font-mono text-[10px] px-2 py-0.5 rounded bg-slate-950 text-cyan-300 border border-cyan-900/60">
                    Expected: {preset.expected_path}
                  </span>
                </button>
              ))}
            </div>
          </div>

          {/* Dynamic Medical Input Form */}
          <form onSubmit={handleSubmitPrediction} className="space-y-8 pt-4 border-t border-slate-800">
            {Object.entries(groupedFeatures).map(([groupName, feats]) => (
              <div key={groupName} className="space-y-3">
                <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider font-mono flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-400"></span>
                  {groupName}
                </h3>

                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                  {feats.map(feat => {
                    const val = featureValues[feat.name] !== undefined ? featureValues[feat.name] : feat.default_value;
                    const isError = !!validationErrors[feat.name];

                    return (
                      <div
                        key={feat.name}
                        className={`bg-slate-900/80 border rounded-2xl p-4 space-y-2 transition-all ${
                          isError ? 'border-rose-500/80 bg-rose-950/20' : 'border-slate-800 focus-within:border-cyan-500/60'
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <label
                            htmlFor={feat.name}
                            className="text-xs font-bold text-slate-200 truncate pr-1"
                            title={feat.description}
                          >
                            {feat.display_name}
                          </label>
                          {feat.unit && (
                            <span className="text-[10px] font-mono text-cyan-400 font-semibold shrink-0">
                              {feat.unit}
                            </span>
                          )}
                        </div>

                        {/* If categorical with options, render select / radio */}
                        {feat.options && feat.options.length > 0 ? (
                          <div className="space-y-1">
                            <select
                              id={feat.name}
                              value={val}
                              onChange={(e) => handleInputChange(feat.name, parseFloat(e.target.value))}
                              className="w-full bg-slate-950 border border-slate-700/90 rounded-xl px-3 py-2 text-xs text-white font-medium focus:outline-none focus:border-cyan-400"
                            >
                              {feat.options.map(opt => (
                                <option key={opt.value} value={opt.value}>
                                  {opt.label}
                                </option>
                              ))}
                            </select>
                          </div>
                        ) : (
                          /* Numerical input */
                          <div className="space-y-1">
                            <input
                              id={feat.name}
                              type="number"
                              step={feat.step || 0.01}
                              min={feat.min_value}
                              max={feat.max_value}
                              value={val}
                              onChange={(e) => handleInputChange(feat.name, parseFloat(e.target.value))}
                              className="w-full bg-slate-950 border border-slate-700/90 rounded-xl px-3 py-2 text-xs text-white font-mono focus:outline-none focus:border-cyan-400"
                            />
                            <div className="flex items-center justify-between text-[10px] font-mono text-slate-500 px-0.5">
                              <span>Min: {feat.min_value}</span>
                              <span>Max: {feat.max_value}</span>
                            </div>
                          </div>
                        )}

                        {isError && (
                          <p className="text-[10px] text-rose-400 font-medium">
                            {validationErrors[feat.name]}
                          </p>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}

            {/* Error banner */}
            {error && (
              <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-xs text-rose-300 flex items-start gap-2.5">
                <AlertCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
                <span>{error}</span>
              </div>
            )}

            {/* Submit Button */}
            <div className="pt-4 border-t border-slate-800 flex items-center justify-end">
              <button
                type="submit"
                disabled={loading}
                className="w-full sm:w-auto px-8 py-3.5 rounded-xl bg-gradient-to-r from-cyan-500 via-blue-600 to-indigo-600 hover:from-cyan-400 hover:to-indigo-500 text-white font-bold text-sm shadow-xl shadow-cyan-500/25 transition-all flex items-center justify-center gap-2.5 disabled:opacity-50"
              >
                <Activity className="w-4 h-4 text-white" />
                Predict Disease
              </button>
            </div>
          </form>
        </section>
      )}

      {/* Step 3: Loading Screen (Analyzing Patient Data...) */}
      {loading && (
        <section className="glass-panel rounded-3xl p-12 text-center border border-cyan-500/30 shadow-2xl space-y-6 animate-pulse">
          <div className="w-16 h-16 rounded-2xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center mx-auto text-cyan-400">
            <Activity className="w-8 h-8 animate-spin" />
          </div>

          <div className="space-y-2">
            <h2 className="text-xl font-black text-white">Analyzing Patient Clinical Data...</h2>
            <p className="text-xs text-slate-400 max-w-md mx-auto">
              Executing PennyLane 4-Qubit simulator with adaptive dynamic uncertainty routing
            </p>
          </div>

          {/* Workflow Pipeline Indicator */}
          <div className="max-w-md mx-auto bg-slate-950/80 p-4 rounded-2xl border border-slate-800 text-xs font-mono text-slate-400 space-y-2 text-left">
            <div className="flex items-center gap-2 text-cyan-300">
              <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping"></span>
              <span>1. Classical Feature Preprocessing (Median + Scale)</span>
            </div>
            <div className="flex items-center gap-2 text-cyan-300">
              <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping"></span>
              <span>2. Preliminary Head & Confidence Calculation</span>
            </div>
            <div className="flex items-center gap-2 text-cyan-300">
              <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping"></span>
              <span>3. Dynamic Quantum Uncertainty Routing</span>
            </div>
            <div className="flex items-center gap-2 text-cyan-300">
              <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping"></span>
              <span>4. Final Predictive Readout & Kernel SHAP</span>
            </div>
          </div>
        </section>
      )}

      {/* Step 4: Full Prediction Result Display */}
      {prediction && !loading && (
        <section className="space-y-8 animate-in fade-in duration-300">
          
          {/* Main Prediction Result Box */}
          <div className="glass-panel rounded-3xl p-8 border border-slate-800 shadow-2xl relative overflow-hidden space-y-6">
            <div className={`absolute top-0 left-0 right-0 h-2 ${
              prediction.prediction_class === 1 ? 'bg-rose-500' : 'bg-emerald-500'
            }`} />

            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-slate-800">
              <div>
                <span className="text-[11px] font-mono uppercase tracking-wider text-cyan-400 font-bold">
                  {prediction.dataset_display_name} Result
                </span>
                <h2 className="text-3xl font-black text-white mt-1 flex items-center gap-3">
                  {prediction.prediction_label}
                  {prediction.prediction_class === 1 ? (
                    <span className="text-xs font-mono font-bold px-3 py-1 rounded-full bg-rose-500/15 text-rose-400 border border-rose-500/30">
                      Positive / Disease Detected
                    </span>
                  ) : (
                    <span className="text-xs font-mono font-bold px-3 py-1 rounded-full bg-emerald-500/15 text-emerald-400 border border-emerald-500/30">
                      Negative / Normal Screening
                    </span>
                  )}
                </h2>
              </div>

              {/* Action Buttons */}
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setPrediction(null)}
                  className="px-4 py-2 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-700 text-xs font-bold text-white transition-all flex items-center gap-1.5"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  Run Another Prediction
                </button>
              </div>
            </div>

            {/* Main Scores Grid */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <div className="bg-slate-900/90 p-5 rounded-2xl border border-slate-800">
                <span className="text-xs font-mono text-slate-400 block">Disease Probability</span>
                <span className={`text-3xl font-black font-mono mt-1 block ${
                  prediction.prediction_class === 1 ? 'text-rose-400' : 'text-emerald-400'
                }`}>
                  {(prediction.probability * 100).toFixed(2)}%
                </span>
                <span className="text-[10px] text-slate-500 block mt-1">Calibrated sigmoid output</span>
              </div>

              <div className="bg-slate-900/90 p-5 rounded-2xl border border-slate-800">
                <span className="text-xs font-mono text-slate-400 block">Model Confidence</span>
                <span className="text-3xl font-black font-mono text-cyan-300 mt-1 block">
                  {(prediction.confidence * 100).toFixed(2)}%
                </span>
                <span className="text-[10px] text-slate-500 block mt-1">max(p, 1 - p)</span>
              </div>

              <div className="bg-slate-900/90 p-5 rounded-2xl border border-slate-800">
                <span className="text-xs font-mono text-slate-400 block">Adaptive Route</span>
                <span className="text-2xl font-black font-mono text-white mt-1 block flex items-center gap-1">
                  <Zap className="w-5 h-5 text-amber-400" />
                  Path {prediction.quantum_path}
                </span>
                <span className="text-[10px] text-slate-400 block mt-1">{prediction.quantum_path_name}</span>
              </div>

              <div className="bg-slate-900/90 p-5 rounded-2xl border border-slate-800">
                <span className="text-xs font-mono text-slate-400 block">Quantum Layers Used</span>
                <span className="text-3xl font-black font-mono text-purple-400 mt-1 block">
                  {prediction.quantum_layers_executed} QL
                </span>
                <span className="text-[10px] text-slate-400 block mt-1">
                  {prediction.quantum_layers_executed === 0 ? 'Quantum bypassed' : `${prediction.quantum_layers_executed} layer(s) executed`}
                </span>
              </div>
            </div>

            {/* Adaptive Routing Flowchart - Only Active Path Highlighted */}
            <div className="bg-slate-900/60 p-6 rounded-2xl border border-slate-800 space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-bold text-white font-mono uppercase tracking-wider flex items-center gap-2">
                  <Cpu className="w-4 h-4 text-cyan-400" />
                  Adaptive Quantum Routing Decision Flow
                </h3>
                <span className="text-[11px] font-mono text-slate-400">
                  Thresholds: τ_high = {(prediction.tau_high * 100).toFixed(0)}% | τ_low = {(prediction.tau_low * 100).toFixed(0)}%
                </span>
              </div>

              {/* 3-Path Visualizer Columns */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
                
                {/* Path 0 */}
                <div className={`p-4 rounded-2xl border transition-all duration-300 ${
                  prediction.quantum_path === 0
                    ? 'bg-emerald-950/60 border-emerald-400 shadow-xl shadow-emerald-500/10 ring-2 ring-emerald-500/40'
                    : 'bg-slate-950/40 border-slate-800/60 opacity-40 grayscale'
                }`}>
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold font-mono text-emerald-400">PATH 0 (0 QL)</span>
                    {prediction.quantum_path === 0 && <CheckCircle2 className="w-4 h-4 text-emerald-400" />}
                  </div>
                  <span className="text-[11px] font-mono text-slate-400 block mt-1">Confidence ≥ 85%</span>
                  <p className="text-xs text-slate-300 mt-2">
                    High diagnostic certainty. Quantum variational layers bypassed to provide lowest latency.
                  </p>
                </div>

                {/* Path 1 */}
                <div className={`p-4 rounded-2xl border transition-all duration-300 ${
                  prediction.quantum_path === 1
                    ? 'bg-amber-950/60 border-amber-400 shadow-xl shadow-amber-500/10 ring-2 ring-amber-500/40'
                    : 'bg-slate-950/40 border-slate-800/60 opacity-40 grayscale'
                }`}>
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold font-mono text-amber-400">PATH 1 (1 QL)</span>
                    {prediction.quantum_path === 1 && <CheckCircle2 className="w-4 h-4 text-amber-400" />}
                  </div>
                  <span className="text-[11px] font-mono text-slate-400 block mt-1">65% ≤ Confidence &lt; 85%</span>
                  <p className="text-xs text-slate-300 mt-2">
                    Intermediate uncertainty. Escalated to 1-layer 4-qubit variational circuit for refinement.
                  </p>
                </div>

                {/* Path 2 */}
                <div className={`p-4 rounded-2xl border transition-all duration-300 ${
                  prediction.quantum_path === 2
                    ? 'bg-purple-950/60 border-purple-400 shadow-xl shadow-purple-500/10 ring-2 ring-purple-500/40'
                    : 'bg-slate-950/40 border-slate-800/60 opacity-40 grayscale'
                }`}>
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold font-mono text-purple-400">PATH 2 (2 QL)</span>
                    {prediction.quantum_path === 2 && <CheckCircle2 className="w-4 h-4 text-purple-400" />}
                  </div>
                  <span className="text-[11px] font-mono text-slate-400 block mt-1">Confidence &lt; 65%</span>
                  <p className="text-xs text-slate-300 mt-2">
                    High diagnostic uncertainty. Escalated to full 2-layer 4-qubit variational circuit.
                  </p>
                </div>

              </div>

              {/* Execution Reason Log */}
              <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 text-xs text-slate-300">
                <strong>Routing Note:</strong> {prediction.routing_reason}
              </div>
            </div>

            {/* Explainability / SHAP Feature Attribution Section */}
            {explanation && (
              <div className="bg-slate-900/60 p-6 rounded-2xl border border-slate-800 space-y-4">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                  <div>
                    <h3 className="text-sm font-bold text-white flex items-center gap-2">
                      <Sparkles className="w-4 h-4 text-cyan-400" />
                      Why Did the Model Make This Prediction? (Kernel SHAP)
                    </h3>
                    <p className="text-xs text-slate-400 mt-0.5">
                      Quantifies individual clinical feature contributions toward the model's prediction boundary
                    </p>
                  </div>
                  <span className="text-[11px] font-mono text-slate-400">
                    Base Probability: {(explanation.base_value * 100).toFixed(1)}%
                  </span>
                </div>

                {/* Horizontal Bar Chart for Top 8 Feature Attributions */}
                <div className="h-64 w-full pt-2">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={explanation.feature_attributions.slice(0, 8).map(f => ({
                        name: f.display_name,
                        value: f.shap_value,
                        direction: f.direction
                      }))}
                      layout="vertical"
                      margin={{ top: 5, right: 30, left: 110, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis type="number" stroke="#94a3b8" fontSize={11} />
                      <YAxis dataKey="name" type="category" stroke="#94a3b8" fontSize={11} width={120} />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#090d16', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }}
                        formatter={(val: any) => [Number(val).toFixed(4), 'SHAP Value']}
                      />
                      <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                        {explanation.feature_attributions.slice(0, 8).map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.direction === 'positive' ? '#f43f5e' : '#10b981'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                {/* Positive vs Negative Factor Breakdown */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
                  <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
                    <span className="text-xs font-bold text-rose-400 font-mono uppercase block mb-2">
                      ▲ Features Increasing Predicted Disease Risk
                    </span>
                    <div className="space-y-1.5 text-xs">
                      {explanation.top_positive_features.map(f => (
                        <div key={f.feature} className="flex justify-between text-slate-300">
                          <span>{f.display_name} ({f.value})</span>
                          <span className="font-mono text-rose-400 font-bold">+{f.shap_value.toFixed(4)}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
                    <span className="text-xs font-bold text-emerald-400 font-mono uppercase block mb-2">
                      ▼ Features Decreasing Predicted Disease Risk
                    </span>
                    <div className="space-y-1.5 text-xs">
                      {explanation.top_negative_features.map(f => (
                        <div key={f.feature} className="flex justify-between text-slate-300">
                          <span>{f.display_name} ({f.value})</span>
                          <span className="font-mono text-emerald-400 font-bold">{f.shap_value.toFixed(4)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Scientific Disclaimer */}
                <p className="text-[11px] text-slate-400 italic pt-2 border-t border-slate-800">
                  {explanation.routing_association_note}
                </p>
              </div>
            )}

            {/* Bottom Actions */}
            <div className="pt-4 border-t border-slate-800 flex flex-wrap items-center justify-between gap-4">
              <button
                type="button"
                onClick={() => setPrediction(null)}
                className="px-6 py-3 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-bold text-xs shadow-lg shadow-cyan-500/20 flex items-center gap-2"
              >
                <RotateCcw className="w-4 h-4" />
                Run Another Prediction
              </button>

              <button
                type="button"
                onClick={() => navigate('/history')}
                className="px-4 py-3 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-700 text-xs font-semibold text-slate-300 flex items-center gap-1.5"
              >
                <History className="w-4 h-4 text-slate-400" />
                View in Prediction History
              </button>
            </div>

          </div>

        </section>
      )}

    </div>
  );
};
