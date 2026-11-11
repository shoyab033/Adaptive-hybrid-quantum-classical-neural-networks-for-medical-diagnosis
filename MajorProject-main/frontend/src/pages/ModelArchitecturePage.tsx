import React from 'react';
import { Layers, Cpu, ShieldCheck, ArrowDown, Zap, FileText, CheckCircle } from 'lucide-react';

export const ModelArchitecturePage: React.FC = () => {
  return (
    <div className="space-y-10 py-4 max-w-5xl mx-auto">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-black text-white tracking-tight flex items-center gap-2">
          <Layers className="w-6 h-6 text-brand-400" />
          Adaptive HQNN Architecture Deep-Dive
        </h1>
        <p className="text-xs text-slate-400 mt-1">
          Detailed mathematical formulation of the dynamic classical encoder, confidence routing, and parameterized variational quantum circuits
        </p>
      </div>

      {/* Architecture Flow Breakdown */}
      <div className="space-y-6">
        
        {/* Module 1: Classical Encoder */}
        <div className="glass-panel rounded-2xl p-6 border border-slate-800 space-y-4">
          <div className="flex items-center gap-2">
            <span className="w-7 h-7 rounded-lg bg-brand-500/20 text-brand-400 font-mono font-bold flex items-center justify-center text-xs">
              01
            </span>
            <h3 className="text-base font-bold text-white">Classical Feature Encoder & Latent Projection</h3>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            The input feature vector x in ℝᴰ is transformed into a dense 16-dimensional intermediate hidden state h(x) via a feedforward layer, layer normalization, and ReLU activation:
          </p>
          <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-800 font-mono text-xs text-brand-300 text-center">
            h(x) = ReLU(LayerNorm(W_e · x + b_e)) ∈ ℝ¹⁶
          </div>
        </div>

        {/* Module 2: Preliminary Prediction & Confidence */}
        <div className="glass-panel rounded-2xl p-6 border border-slate-800 space-y-4">
          <div className="flex items-center gap-2">
            <span className="w-7 h-7 rounded-lg bg-brand-500/20 text-brand-400 font-mono font-bold flex items-center justify-center text-xs">
              02
            </span>
            <h3 className="text-base font-bold text-white">Preliminary Classification & Confidence Estimation</h3>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            A linear classification head produces preliminary logit z_prelim(x) and logistic probability p_prelim(x). Prediction confidence C(x) in [0.5, 1.0] measures the distance from the decision boundary:
          </p>
          <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-800 font-mono text-xs text-brand-300 text-center">
            C(x) = max(p_prelim(x), 1.0 - p_prelim(x))
          </div>
        </div>

        {/* Module 3: Adaptive Uncertainty Router */}
        <div className="glass-panel rounded-2xl p-6 border border-brand-500/30 space-y-4 shadow-lg shadow-brand-500/5">
          <div className="flex items-center gap-2">
            <span className="w-7 h-7 rounded-lg bg-amber-500/20 text-amber-400 font-mono font-bold flex items-center justify-center text-xs">
              03
            </span>
            <h3 className="text-base font-bold text-white">Adaptive Uncertainty Router & Path Assignment</h3>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            Conditioned on confidence C(x) and two calibrated thresholds (τ_high, τ_low) tuned via inner 3-fold cross-validation:
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-2">
            <div className="bg-slate-950 p-3.5 rounded-xl border border-emerald-500/30 text-xs">
              <span className="font-bold text-emerald-400 font-mono block">Path 0 (0 Q-Layers)</span>
              <span className="font-mono text-[11px] text-slate-400 block mt-1">C(x) ≥ τ_high</span>
              <p className="text-[11px] text-slate-300 mt-2">Bypasses quantum circuits. Output is z_prelim directly.</p>
            </div>

            <div className="bg-slate-950 p-3.5 rounded-xl border border-amber-500/30 text-xs">
              <span className="font-bold text-amber-400 font-mono block">Path 1 (1 Q-Layer)</span>
              <span className="font-mono text-[11px] text-slate-400 block mt-1">τ_low ≤ C(x) &lt; τ_high</span>
              <p className="text-[11px] text-slate-300 mt-2">Projects to 4 qubits, executes 1 variational layer.</p>
            </div>

            <div className="bg-slate-950 p-3.5 rounded-xl border border-purple-500/30 text-xs">
              <span className="font-bold text-purple-400 font-mono block">Path 2 (2 Q-Layers)</span>
              <span className="font-mono text-[11px] text-slate-400 block mt-1">C(x) &lt; τ_low</span>
              <p className="text-[11px] text-slate-300 mt-2">Executes full 2-layer variational circuit.</p>
            </div>
          </div>
        </div>

        {/* Module 4: Parameterized Quantum Variational Layers */}
        <div className="glass-panel rounded-2xl p-6 border border-slate-800 space-y-4">
          <div className="flex items-center gap-2">
            <span className="w-7 h-7 rounded-lg bg-purple-500/20 text-purple-400 font-mono font-bold flex items-center justify-center text-xs">
              04
            </span>
            <h3 className="text-base font-bold text-white">Parameterized Variational Quantum Circuit (VQC)</h3>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            The 16-dimensional feature h(x) is projected into a 4-dimensional angle space u(x) = π · tanh(W_u · h(x) + b_u) in [-π, π]⁴. The state is prepared via RY angle embedding on a 4-qubit register:
          </p>
          <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-800 font-mono text-xs text-purple-300 text-center">
            |ψ₀(u)⟩ = ⊗ R_Y(u_j) |0⟩
          </div>
          <p className="text-xs text-slate-300 leading-relaxed mt-2">
            Each variational layer applies single-qubit Euler rotations Rot(α, β, γ) on all 4 qubits followed by circular CNOT entanglers. Pauli-Z expectations are measured and fed to dedicated readout heads (q1_head and q2_head).
          </p>
        </div>

        {/* Module 5: Leakage Prevention Protocol */}
        <div className="glass-panel rounded-2xl p-6 border border-slate-800 space-y-4">
          <div className="flex items-center gap-2">
            <span className="w-7 h-7 rounded-lg bg-emerald-500/20 text-emerald-400 font-mono font-bold flex items-center justify-center text-xs">
              05
            </span>
            <h3 className="text-base font-bold text-white">Zero Data Leakage: Nested 3-Fold Inner Cross-Validation</h3>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            Routing decision thresholds (τ_high, τ_low) are optimized strictly over an inner 3-fold cross-validation grid search inside each outer training fold. The outer validation fold and the isolated 20% holdout test set remain untouched until final evaluation.
          </p>
        </div>

      </div>
    </div>
  );
};
