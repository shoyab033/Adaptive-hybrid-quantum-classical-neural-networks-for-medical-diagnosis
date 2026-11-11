import React from 'react';
import { PredictResponse } from '../../types';
import { Cpu, ArrowDown, Layers, Zap, CheckCircle2, AlertCircle } from 'lucide-react';

interface Props {
  prediction: PredictResponse | null;
  loading?: boolean;
}

export const AdaptiveRoutingVisualizer: React.FC<Props> = ({ prediction, loading }) => {
  const activePath = prediction ? prediction.quantum_path : null;

  return (
    <div className="glass-panel rounded-2xl p-6 border border-slate-800 shadow-xl">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <Cpu className="w-5 h-5 text-brand-400" />
            Adaptive Quantum Routing Flowchart
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Dynamic execution branching based on preliminary classical prediction uncertainty
          </p>
        </div>
        {prediction && (
          <div className={`px-3 py-1 rounded-full text-xs font-mono font-bold flex items-center gap-1.5 border ${
            activePath === 0 
              ? 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30'
              : activePath === 1
              ? 'bg-amber-500/10 text-amber-300 border-amber-500/30'
              : 'bg-purple-500/10 text-purple-300 border-purple-500/30'
          }`}>
            <Zap className="w-3.5 h-3.5" />
            Active Route: Path {activePath} ({prediction.quantum_layers_executed} Q-Layers)
          </div>
        )}
      </div>

      {/* Pipeline Diagram */}
      <div className="flex flex-col items-center space-y-4 max-w-2xl mx-auto">
        
        {/* Step 1: Input Vector */}
        <div className="w-full max-w-md bg-slate-900 border border-slate-700/80 rounded-xl p-3 text-center">
          <div className="text-[11px] font-mono text-slate-400 uppercase tracking-wider">Step 1: Input Medical Vector</div>
          <div className="text-xs font-semibold text-slate-200 mt-0.5">
            {prediction ? `${prediction.dataset_display_name} Feature Record (x ∈ ℝᴰ)` : 'Patient Clinical Features'}
          </div>
        </div>

        <ArrowDown className="w-4 h-4 text-slate-500" />

        {/* Step 2: Classical Feature Encoder */}
        <div className="w-full max-w-md bg-slate-900/90 border border-slate-700/80 rounded-xl p-3.5 text-center">
          <div className="text-[11px] font-mono text-brand-400 uppercase tracking-wider">Step 2: Classical Feature Encoder</div>
          <div className="text-xs text-slate-300 font-mono mt-1">Linear(D → 16) → LayerNorm(16) → ReLU()</div>
          <div className="text-[11px] text-slate-400 mt-1">Generates 16-dimensional intermediate latent representation h(x)</div>
        </div>

        <ArrowDown className="w-4 h-4 text-slate-500" />

        {/* Step 3: Preliminary Head & Confidence */}
        <div className="w-full max-w-lg bg-slate-900/90 border border-slate-700/80 rounded-xl p-4">
          <div className="text-[11px] font-mono text-brand-400 uppercase tracking-wider text-center">
            Step 3: Preliminary Head & Confidence Calculation
          </div>
          <div className="grid grid-cols-2 gap-3 mt-3">
            <div className="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800 text-center">
              <div className="text-[10px] text-slate-400 font-mono">Prelim Probability p_prelim</div>
              <div className="text-sm font-bold text-white font-mono mt-0.5">
                {prediction ? `${(prediction.preliminary_probability * 100).toFixed(1)}%` : '—'}
              </div>
              <div className="text-[10px] text-slate-400 mt-0.5">
                {prediction ? prediction.preliminary_label : 'Probability'}
              </div>
            </div>
            <div className="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800 text-center">
              <div className="text-[10px] text-slate-400 font-mono">Confidence C(x)</div>
              <div className="text-sm font-bold text-brand-300 font-mono mt-0.5">
                {prediction ? `${(prediction.preliminary_confidence * 100).toFixed(1)}%` : '—'}
              </div>
              <div className="text-[10px] text-slate-400 mt-0.5">max(p, 1-p)</div>
            </div>
          </div>
        </div>

        <ArrowDown className="w-4 h-4 text-slate-500" />

        {/* Step 4: Adaptive Router */}
        <div className="w-full max-w-lg bg-slate-900 border border-brand-500/40 rounded-xl p-3.5 text-center shadow-lg shadow-brand-500/5">
          <div className="text-[11px] font-mono text-brand-300 uppercase tracking-wider font-bold">
            Step 4: Adaptive Uncertainty Router
          </div>
          <div className="text-xs text-slate-300 mt-1">
            Thresholds: <span className="font-mono text-emerald-400">τ_high = {prediction ? (prediction.tau_high * 100).toFixed(0) : '85'}%</span> &nbsp;|&nbsp; <span className="font-mono text-purple-400">τ_low = {prediction ? (prediction.tau_low * 100).toFixed(0) : '65'}%</span>
          </div>
        </div>

        {/* 3-Path Branching Columns */}
        <div className="w-full grid grid-cols-1 md:grid-cols-3 gap-3 pt-2">
          
          {/* Path 0 Card */}
          <div className={`rounded-xl p-4 border transition-all duration-300 ${
            activePath === 0
              ? 'bg-emerald-950/40 border-emerald-400 shadow-lg shadow-emerald-500/10 ring-1 ring-emerald-400'
              : 'bg-slate-900/60 border-slate-800 opacity-60'
          }`}>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-bold text-emerald-400 font-mono">PATH 0 (0 QL)</span>
              {activePath === 0 && <CheckCircle2 className="w-4 h-4 text-emerald-400" />}
            </div>
            <div className="text-[11px] font-mono text-slate-400 mb-2">Condition: C(x) ≥ τ_high</div>
            <div className="p-2.5 rounded bg-slate-950/80 border border-slate-800/80 text-[11px] text-slate-300 space-y-1">
              <div>• <strong>0 Quantum Layers</strong></div>
              <div>• Quantum circuits bypassed</div>
              <div>• Direct Classical Logit Output</div>
            </div>
            <div className="mt-2 text-[10px] text-emerald-300 font-medium">
              ⚡ Lowest latency & zero QPU overhead
            </div>
          </div>

          {/* Path 1 Card */}
          <div className={`rounded-xl p-4 border transition-all duration-300 ${
            activePath === 1
              ? 'bg-amber-950/40 border-amber-400 shadow-lg shadow-amber-500/10 ring-1 ring-amber-400'
              : 'bg-slate-900/60 border-slate-800 opacity-60'
          }`}>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-bold text-amber-400 font-mono">PATH 1 (1 QL)</span>
              {activePath === 1 && <CheckCircle2 className="w-4 h-4 text-amber-400" />}
            </div>
            <div className="text-[11px] font-mono text-slate-400 mb-2">Condition: τ_low ≤ C(x) &lt; τ_high</div>
            <div className="p-2.5 rounded bg-slate-950/80 border border-slate-800/80 text-[11px] text-slate-300 space-y-1">
              <div>• <strong>1 Quantum Layer</strong> (4 Qubits)</div>
              <div>• RY Angle Embedding</div>
              <div>• 4 Rot + Circular CNOTs</div>
              <div>• Pauli-Z Expectation Readout</div>
            </div>
            <div className="mt-2 text-[10px] text-amber-300 font-medium">
              🔬 Single-layer quantum refinement
            </div>
          </div>

          {/* Path 2 Card */}
          <div className={`rounded-xl p-4 border transition-all duration-300 ${
            activePath === 2
              ? 'bg-purple-950/40 border-purple-400 shadow-lg shadow-purple-500/10 ring-1 ring-purple-400'
              : 'bg-slate-900/60 border-slate-800 opacity-60'
          }`}>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-bold text-purple-400 font-mono">PATH 2 (2 QL)</span>
              {activePath === 2 && <CheckCircle2 className="w-4 h-4 text-purple-400" />}
            </div>
            <div className="text-[11px] font-mono text-slate-400 mb-2">Condition: C(x) &lt; τ_low</div>
            <div className="p-2.5 rounded bg-slate-950/80 border border-slate-800/80 text-[11px] text-slate-300 space-y-1">
              <div>• <strong>2 Quantum Layers</strong> (4 Qubits)</div>
              <div>• RY Angle Embedding</div>
              <div>• 8 Rot + 8 Circular CNOTs</div>
              <div>• Pauli-Z Expectation Readout</div>
            </div>
            <div className="mt-2 text-[10px] text-purple-300 font-medium">
              🌌 Deep quantum transformation for uncertain cases
            </div>
          </div>

        </div>

        {/* Dynamic Explanation Note */}
        {prediction && (
          <div className="w-full bg-slate-900/90 border border-slate-800 rounded-xl p-4 mt-2">
            <div className="flex items-start gap-2.5">
              <AlertCircle className="w-4 h-4 text-brand-400 mt-0.5 shrink-0" />
              <div className="text-xs text-slate-300 leading-relaxed">
                <strong>Execution Log:</strong> {prediction.routing_reason}
              </div>
            </div>
          </div>
        )}

      </div>
    </div>
  );
};
