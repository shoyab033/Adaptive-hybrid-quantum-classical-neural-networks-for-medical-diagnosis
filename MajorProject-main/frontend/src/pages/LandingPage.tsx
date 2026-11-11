import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { getModels } from '../services/api';
import { DatasetMetadata } from '../types';
import { 
  Activity, 
  ArrowRight, 
  Cpu, 
  Stethoscope, 
  Heart, 
  Droplet, 
  ShieldCheck, 
  Sparkles,
  Info,
  CheckCircle2,
  Lock,
  Layers
} from 'lucide-react';

export const LandingPage: React.FC = () => {
  const [models, setModels] = useState<DatasetMetadata[]>([]);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchModels = async () => {
      try {
        const data = await getModels();
        setModels(data);
      } catch (e) {
        console.error('Error loading models', e);
      }
    };
    fetchModels();
  }, []);

  const getDiseaseIcon = (key: string) => {
    if (key.includes('heart')) return Heart;
    if (key.includes('diabetes')) return Droplet;
    return ShieldCheck;
  };

  const getDiseaseColor = (key: string) => {
    if (key.includes('heart')) return 'from-rose-500/20 to-rose-600/10 border-rose-500/30 text-rose-400 hover:border-rose-500/60';
    if (key.includes('diabetes')) return 'from-amber-500/20 to-amber-600/10 border-amber-500/30 text-amber-400 hover:border-amber-500/60';
    return 'from-cyan-500/20 to-cyan-600/10 border-cyan-500/30 text-cyan-400 hover:border-cyan-500/60';
  };

  return (
    <div className="space-y-16 py-6 max-w-6xl mx-auto">
      
      {/* Hero Section */}
      <section className="relative overflow-hidden rounded-3xl bg-gradient-to-b from-slate-900/90 via-slate-950 to-slate-950 border border-slate-800/90 p-8 sm:p-14 shadow-2xl">
        <div className="absolute -top-28 -right-28 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute -bottom-28 -left-28 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl pointer-events-none" />

        <div className="relative z-10 max-w-3xl space-y-6">
          <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-xs font-mono font-semibold text-cyan-300">
            <Stethoscope className="w-4 h-4 text-cyan-400" />
            Adaptive HQNN Medical Disease Prediction
          </div>

          <h1 className="text-3xl sm:text-5xl font-black text-white tracking-tight leading-tight">
            Adaptive HQNN <br />
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 via-blue-400 to-indigo-400">
              Medical Disease Prediction
            </span>
          </h1>

          <p className="text-sm sm:text-base text-slate-300 leading-relaxed font-normal max-w-2xl">
            An adaptive hybrid quantum-classical neural network for research-based medical disease prediction. Input patient measurements to run real-time inference across oncology, cardiology, and endocrinology models with dynamic quantum layer allocation.
          </p>

          <div className="flex flex-wrap items-center gap-4 pt-2">
            <Link
              to="/predict"
              className="px-7 py-3.5 rounded-xl bg-gradient-to-r from-cyan-500 via-blue-600 to-indigo-600 hover:from-cyan-400 hover:to-indigo-500 text-white font-bold text-sm shadow-xl shadow-cyan-500/25 transition-all flex items-center gap-2 hover:scale-[1.02]"
            >
              <Activity className="w-4 h-4 text-cyan-200" />
              Start Prediction
              <ArrowRight className="w-4 h-4" />
            </Link>

            <Link
              to="/how-it-works"
              className="px-6 py-3.5 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-700 text-white font-semibold text-sm transition-all flex items-center gap-2"
            >
              <Info className="w-4 h-4 text-slate-400" />
              How It Works
            </Link>
          </div>
        </div>

        {/* Clear Medical Disclaimer */}
        <div className="mt-10 pt-6 border-t border-slate-800/80 text-xs text-amber-300/80 bg-amber-500/5 p-4 rounded-2xl border border-amber-500/20 flex items-start gap-2.5">
          <Info className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
          <p>
            <strong>Research Prototype:</strong> This system is intended for educational and research purposes only and is not a substitute for professional medical diagnosis.
          </p>
        </div>
      </section>

      {/* Select Disease Prediction Task Section */}
      <section className="space-y-6">
        <div className="text-center max-w-2xl mx-auto space-y-2">
          <h2 className="text-2xl font-black text-white tracking-tight">Select a Disease Prediction Task</h2>
          <p className="text-xs text-slate-400">
            Choose a clinical domain to launch the interactive medical input form and evaluate real patient data
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {models.map((model) => {
            const Icon = getDiseaseIcon(model.key);
            const colorClass = getDiseaseColor(model.key);

            return (
              <div
                key={model.key}
                className={`glass-panel rounded-2xl p-6 border transition-all duration-200 flex flex-col justify-between space-y-5 bg-gradient-to-b ${colorClass}`}
              >
                <div className="space-y-3">
                  <div className="w-12 h-12 rounded-xl bg-slate-900/90 border border-slate-700/80 flex items-center justify-center shadow">
                    <Icon className="w-6 h-6" />
                  </div>

                  <div>
                    <h3 className="text-lg font-bold text-white">{model.short_name || model.display_name}</h3>
                    <div className="text-[11px] font-mono text-slate-400">{model.domain.split('(')[0]}</div>
                  </div>

                  <p className="text-xs text-slate-300 leading-relaxed line-clamp-3">
                    {model.description}
                  </p>
                </div>

                <div className="space-y-4 pt-3 border-t border-slate-800/80">
                  <div className="flex items-center justify-between text-xs font-mono text-slate-400">
                    <span>Required Features:</span>
                    <strong className="text-white">{model.num_features} measurements</strong>
                  </div>

                  <button
                    onClick={() => navigate(`/predict?model=${model.key}`)}
                    className="w-full py-2.5 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-700 text-xs font-bold text-white transition-all flex items-center justify-center gap-2 shadow hover:border-slate-500"
                  >
                    Select {model.short_name || 'Task'}
                    <ArrowRight className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* Simple 3-Step Workflow Section */}
      <section className="glass-panel rounded-3xl p-8 sm:p-10 border border-slate-800/80 space-y-8">
        <div className="text-center max-w-xl mx-auto space-y-1">
          <h3 className="text-xl font-bold text-white">How the Prediction System Operates</h3>
          <p className="text-xs text-slate-400">From patient clinical parameters to explainable diagnostic decision</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-slate-900/80 p-5 rounded-2xl border border-slate-800 space-y-2">
            <div className="w-8 h-8 rounded-lg bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 font-mono font-bold flex items-center justify-center text-xs">
              01
            </div>
            <h4 className="text-sm font-bold text-white">Enter Clinical Features</h4>
            <p className="text-xs text-slate-400 leading-relaxed">
              Input patient metrics (e.g. glucose, blood pressure, ECG, or cell morphology) or load one of our calibrated clinical sample presets.
            </p>
          </div>

          <div className="bg-slate-900/80 p-5 rounded-2xl border border-slate-800 space-y-2">
            <div className="w-8 h-8 rounded-lg bg-blue-500/10 border border-blue-500/30 text-blue-400 font-mono font-bold flex items-center justify-center text-xs">
              02
            </div>
            <h4 className="text-sm font-bold text-white">Adaptive Quantum Routing</h4>
            <p className="text-xs text-slate-400 leading-relaxed">
              The model computes preliminary classical uncertainty and dynamically allocates 0, 1, or 2 quantum variational layers for optimal accuracy and speed.
            </p>
          </div>

          <div className="bg-slate-900/80 p-5 rounded-2xl border border-slate-800 space-y-2">
            <div className="w-8 h-8 rounded-lg bg-indigo-500/10 border border-indigo-500/30 text-indigo-400 font-mono font-bold flex items-center justify-center text-xs">
              03
            </div>
            <h4 className="text-sm font-bold text-white">Result & SHAP Explanation</h4>
            <p className="text-xs text-slate-400 leading-relaxed">
              View the diagnostic prediction, confidence score, actual quantum layers used, and feature contributions that influenced the prediction.
            </p>
          </div>
        </div>
      </section>

    </div>
  );
};
