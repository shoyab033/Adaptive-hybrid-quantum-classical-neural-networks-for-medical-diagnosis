import React from 'react';
import { Link } from 'react-router-dom';
import { 
  Info, 
  Cpu, 
  Activity, 
  Zap, 
  Layers, 
  ShieldCheck, 
  Sparkles, 
  ArrowRight,
  TrendingDown
} from 'lucide-react';

export const HowItWorksPage: React.FC = () => {
  return (
    <div className="space-y-12 py-4 max-w-4xl mx-auto">
      {/* Header */}
      <div className="text-center space-y-3">
        <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-xs font-mono text-cyan-300">
          <Info className="w-4 h-4 text-cyan-400" />
          Plain English System Architecture
        </div>
        <h1 className="text-3xl sm:text-4xl font-black text-white tracking-tight">
          How Dynamic Quantum Layer Selection Works
        </h1>
        <p className="text-sm text-slate-300 max-w-2xl mx-auto leading-relaxed">
          Discover how the Adaptive HQNN intelligently balances classical compute efficiency with quantum expressive power during medical disease screening.
        </p>
      </div>

      {/* The Core Motivation */}
      <div className="glass-panel rounded-3xl p-8 border border-slate-800 space-y-4 shadow-xl">
        <h2 className="text-lg font-bold text-white flex items-center gap-2">
          <Cpu className="w-5 h-5 text-cyan-400" />
          The Limitation of Conventional Quantum Machine Learning
        </h2>
        <p className="text-xs text-slate-300 leading-relaxed">
          Standard Hybrid Quantum-Classical Neural Networks (HQNNs) execute the exact same fixed-depth parameterized quantum circuit on <strong>every single patient sample</strong>. In clinical practice, the vast majority of patient profiles are straightforward (e.g. clearly healthy or severe symptomatic cases) and do not need expensive quantum circuit evaluations. Running full quantum circuits for routine cases wastes computational resources and increases inference latency.
        </p>
      </div>

      {/* The Adaptive Routing Solution */}
      <div className="space-y-6">
        <div className="text-center space-y-1">
          <h2 className="text-xl font-bold text-white">Three Calibrated Execution Pathways</h2>
          <p className="text-xs text-slate-400">The model routes each patient sample based on its preliminary classical confidence</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="glass-panel rounded-2xl p-6 border border-emerald-500/30 space-y-3">
            <div className="w-10 h-10 rounded-xl bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center font-mono font-bold text-emerald-400">
              0
            </div>
            <h3 className="text-sm font-bold text-white">Path 0: Classical Only</h3>
            <span className="text-[11px] font-mono text-emerald-400 block font-semibold">Confidence ≥ 85%</span>
            <p className="text-xs text-slate-300 leading-relaxed">
              When the initial classical encoder produces high diagnostic certainty, quantum circuit execution is skipped entirely. <strong>Zero quantum layers</strong> are executed.
            </p>
          </div>

          <div className="glass-panel rounded-2xl p-6 border border-amber-500/30 space-y-3">
            <div className="w-10 h-10 rounded-xl bg-amber-500/10 border border-amber-500/30 flex items-center justify-center font-mono font-bold text-amber-400">
              1
            </div>
            <h3 className="text-sm font-bold text-white">Path 1: Single Quantum Layer</h3>
            <span className="text-[11px] font-mono text-amber-400 block font-semibold">65% ≤ Confidence &lt; 85%</span>
            <p className="text-xs text-slate-300 leading-relaxed">
              For samples presenting moderate diagnostic ambiguity, latent features are projected to a 4-qubit register and evaluated through a <strong>1-layer variational quantum circuit</strong>.
            </p>
          </div>

          <div className="glass-panel rounded-2xl p-6 border border-purple-500/30 space-y-3">
            <div className="w-10 h-10 rounded-xl bg-purple-500/10 border border-purple-500/30 flex items-center justify-center font-mono font-bold text-purple-400">
              2
            </div>
            <h3 className="text-sm font-bold text-white">Path 2: Full Two Quantum Layers</h3>
            <span className="text-[11px] font-mono text-purple-400 block font-semibold">Confidence &lt; 65%</span>
            <p className="text-xs text-slate-300 leading-relaxed">
              Borderline patient presentations near the diagnostic threshold escalate to the full <strong>depth-2 variational quantum circuit</strong> (8 Rotations + 8 CNOTs) for maximum feature expressivity.
            </p>
          </div>
        </div>
      </div>

      {/* Quantum Workload Savings */}
      <div className="glass-panel rounded-3xl p-8 border border-slate-800 space-y-6">
        <h2 className="text-lg font-bold text-white flex items-center gap-2">
          <TrendingDown className="w-5 h-5 text-emerald-400" />
          Empirical Inference Workload Reductions
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-center">
          <div className="bg-slate-900/90 p-4 rounded-2xl border border-slate-800">
            <div className="text-3xl font-black font-mono text-emerald-400">98.8%</div>
            <div className="text-xs font-bold text-white mt-1">WDBC Breast Cancer</div>
            <p className="text-[11px] text-slate-400 mt-0.5">0.02 vs 2.00 Q-layers / sample</p>
          </div>

          <div className="bg-slate-900/90 p-4 rounded-2xl border border-slate-800">
            <div className="text-3xl font-black font-mono text-cyan-400">87.8%</div>
            <div className="text-xs font-bold text-white mt-1">Cleveland Heart Disease</div>
            <p className="text-[11px] text-slate-400 mt-0.5">0.24 vs 2.00 Q-layers / sample</p>
          </div>

          <div className="bg-slate-900/90 p-4 rounded-2xl border border-slate-800">
            <div className="text-3xl font-black font-mono text-purple-400">79.6%</div>
            <div className="text-xs font-bold text-white mt-1">Pima Indians Diabetes</div>
            <p className="text-[11px] text-slate-400 mt-0.5">0.41 vs 2.00 Q-layers / sample</p>
          </div>
        </div>
      </div>

      {/* CTA Box */}
      <div className="p-8 rounded-3xl bg-gradient-to-r from-cyan-950/80 via-blue-950/80 to-indigo-950/80 border border-cyan-500/30 flex flex-col sm:flex-row items-center justify-between gap-6 shadow-2xl">
        <div>
          <h3 className="text-lg font-bold text-white">Ready to Run a Prediction?</h3>
          <p className="text-xs text-slate-300 mt-1">
            Test the live Adaptive HQNN model across diabetes, heart disease, or breast cancer.
          </p>
        </div>
        <Link
          to="/predict"
          className="px-6 py-3.5 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white font-bold text-xs shadow-lg shadow-cyan-500/25 transition-all flex items-center gap-2 shrink-0"
        >
          <Activity className="w-4 h-4" />
          Go to Predict Disease
          <ArrowRight className="w-4 h-4" />
        </Link>
      </div>

    </div>
  );
};
