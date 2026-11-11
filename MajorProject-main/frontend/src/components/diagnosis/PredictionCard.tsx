import React from 'react';
import { PredictResponse } from '../../types';
import { 
  CheckCircle, 
  AlertTriangle, 
  Cpu, 
  Zap, 
  ShieldAlert, 
  Clock, 
  Layers,
  ArrowRight,
  Sparkles
} from 'lucide-react';

interface Props {
  prediction: PredictResponse;
  onExplainClick?: () => void;
}

export const PredictionCard: React.FC<Props> = ({ prediction, onExplainClick }) => {
  const isPositive = prediction.prediction_class === 1;
  const probPercent = (prediction.probability * 100).toFixed(1);
  const confPercent = (prediction.confidence * 100).toFixed(1);

  return (
    <div className="glass-panel rounded-2xl p-6 border border-slate-800 shadow-2xl relative overflow-hidden">
      {/* Top Background Glow */}
      <div className={`absolute top-0 left-0 right-0 h-1.5 ${
        isPositive ? 'bg-rose-500' : 'bg-emerald-500'
      }`} />

      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-5 border-b border-slate-800/80">
        <div>
          <span className="text-[11px] font-mono uppercase tracking-wider text-slate-400">
            {prediction.dataset_display_name}
          </span>
          <h2 className="text-2xl font-extrabold text-white flex items-center gap-2 mt-0.5">
            {prediction.prediction_label}
            {isPositive ? (
              <span className="text-xs font-mono font-bold px-2.5 py-0.5 rounded-full bg-rose-500/10 text-rose-400 border border-rose-500/30">
                Positive Screening
              </span>
            ) : (
              <span className="text-xs font-mono font-bold px-2.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                Negative / Normal
              </span>
            )}
          </h2>
        </div>

        {/* Explain Button */}
        {onExplainClick && (
          <button
            onClick={onExplainClick}
            className="self-start md:self-auto px-4 py-2 rounded-xl bg-brand-600/20 hover:bg-brand-600/30 border border-brand-500/40 text-xs font-semibold text-brand-300 transition-all flex items-center gap-1.5 shadow-sm"
          >
            <Sparkles className="w-4 h-4 text-brand-400" />
            View SHAP Attributions
          </button>
        )}
      </div>

      {/* Main Metrics Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 my-6">
        <div className="bg-slate-900/90 p-4 rounded-xl border border-slate-800">
          <span className="text-[11px] text-slate-400 font-mono block">Disease Probability</span>
          <span className={`text-2xl font-black font-mono mt-1 block ${
            isPositive ? 'text-rose-400' : 'text-emerald-400'
          }`}>
            {probPercent}%
          </span>
          <span className="text-[10px] text-slate-400 block mt-1">Sigmoid Logit Output</span>
        </div>

        <div className="bg-slate-900/90 p-4 rounded-xl border border-slate-800">
          <span className="text-[11px] text-slate-400 font-mono block">Decision Confidence</span>
          <span className="text-2xl font-black font-mono text-brand-400 mt-1 block">
            {confPercent}%
          </span>
          <span className="text-[10px] text-slate-400 block mt-1">max(p, 1-p)</span>
        </div>

        <div className="bg-slate-900/90 p-4 rounded-xl border border-slate-800">
          <span className="text-[11px] text-slate-400 font-mono block">Dynamic Routing Path</span>
          <span className="text-lg font-bold font-mono text-white mt-1 block flex items-center gap-1">
            <Zap className="w-4 h-4 text-amber-400" />
            Path {prediction.quantum_path}
          </span>
          <span className="text-[10px] text-slate-400 block mt-1">{prediction.quantum_path_name}</span>
        </div>

        <div className="bg-slate-900/90 p-4 rounded-xl border border-slate-800">
          <span className="text-[11px] text-slate-400 font-mono block">Quantum Layers Executed</span>
          <span className="text-2xl font-black font-mono text-purple-400 mt-1 block">
            {prediction.quantum_layers_executed} QL
          </span>
          <span className="text-[10px] text-slate-400 block mt-1">
            {prediction.quantum_layers_executed === 0 ? '100% QPU Bypassed' : prediction.quantum_layers_executed === 1 ? '50% QPU Savings' : 'Full 2-Layer Depth'}
          </span>
        </div>
      </div>

      {/* Preliminary vs Final Prediction Comparison Box */}
      <div className="bg-slate-900/60 rounded-xl p-4 border border-slate-800/80">
        <h4 className="text-xs font-bold text-slate-300 font-mono uppercase tracking-wider mb-3 flex items-center gap-1.5">
          <Layers className="w-3.5 h-3.5 text-brand-400" />
          Multi-Stage Classical & Quantum Execution Summary
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          <div className="p-3 rounded-lg bg-slate-950/80 border border-slate-800">
            <span className="text-slate-400 font-medium">Stage 1: Preliminary Classical Head</span>
            <div className="mt-1.5 flex items-center justify-between">
              <span className="font-mono text-slate-200">Probability: {(prediction.preliminary_probability * 100).toFixed(1)}%</span>
              <span className="font-mono text-brand-400 font-bold">Confidence: {(prediction.preliminary_confidence * 100).toFixed(1)}%</span>
            </div>
          </div>

          <div className="p-3 rounded-lg bg-slate-950/80 border border-slate-800">
            <span className="text-slate-400 font-medium">Stage 2: Final Calibrated Prediction</span>
            <div className="mt-1.5 flex items-center justify-between">
              <span className="font-mono text-slate-200">Route: Path {prediction.quantum_path} ({prediction.quantum_layers_executed} QL)</span>
              <span className={`font-mono font-bold ${isPositive ? 'text-rose-400' : 'text-emerald-400'}`}>
                {prediction.prediction_label}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
