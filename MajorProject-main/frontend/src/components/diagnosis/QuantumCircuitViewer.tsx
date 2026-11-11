import React, { useState } from 'react';
import { Cpu, Eye, Info, Sparkles } from 'lucide-react';

interface Props {
  selectedPath?: number; // 0, 1, or 2
  pauliExpectations?: number[] | null;
  latentVector?: number[] | null;
}

export const QuantumCircuitViewer: React.FC<Props> = ({
  selectedPath = 1,
  pauliExpectations = null,
  latentVector = null,
}) => {
  const [activePathView, setActivePathView] = useState<number>(selectedPath);

  // Sync if prop updates
  React.useEffect(() => {
    if (selectedPath !== undefined && selectedPath !== null) {
      setActivePathView(selectedPath);
    }
  }, [selectedPath]);

  const qubits = [0, 1, 2, 3];

  return (
    <div className="glass-panel rounded-2xl p-6 border border-slate-800 shadow-xl">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
        <div>
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <Cpu className="w-5 h-5 text-quantum-400" />
            4-Qubit Parameterized Variational Quantum Circuit
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Gate-level circuit structure with RY angle embedding, single-qubit Euler rotations, and circular CNOT entanglement
          </p>
        </div>

        {/* Path Selector Tabs */}
        <div className="flex items-center bg-slate-900 border border-slate-800 p-1 rounded-xl">
          <button
            onClick={() => setActivePathView(0)}
            className={`px-3 py-1 rounded-lg text-xs font-mono font-medium transition-all ${
              activePathView === 0
                ? 'bg-emerald-500 text-slate-950 font-bold shadow'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            Path 0 (0 QL)
          </button>
          <button
            onClick={() => setActivePathView(1)}
            className={`px-3 py-1 rounded-lg text-xs font-mono font-medium transition-all ${
              activePathView === 1
                ? 'bg-amber-500 text-slate-950 font-bold shadow'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            Path 1 (1 QL)
          </button>
          <button
            onClick={() => setActivePathView(2)}
            className={`px-3 py-1 rounded-lg text-xs font-mono font-medium transition-all ${
              activePathView === 2
                ? 'bg-purple-500 text-slate-950 font-bold shadow'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            Path 2 (2 QL)
          </button>
        </div>
      </div>

      {/* Path 0 (Bypassed) State */}
      {activePathView === 0 ? (
        <div className="bg-slate-900/60 border border-emerald-500/20 rounded-xl p-8 text-center my-4">
          <div className="w-12 h-12 rounded-full bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center mx-auto mb-3">
            <Eye className="w-6 h-6 text-emerald-400" />
          </div>
          <h4 className="text-sm font-bold text-white">Path 0: Quantum Circuit Execution Bypassed</h4>
          <p className="text-xs text-slate-400 max-w-md mx-auto mt-1 leading-relaxed">
            Because the classical feature encoder produced high confidence (C(x) ≥ τ_high), the input is resolved strictly via the classical logit head (z_prelim). Zero quantum gates, zero state preparations, and zero measurements are executed.
          </p>
          <div className="mt-4 inline-flex items-center gap-1.5 px-3 py-1 rounded bg-emerald-950/60 border border-emerald-800/60 text-xs font-mono text-emerald-300">
            Quantum Layers Executed = 0 &nbsp;|&nbsp; Latency = Minimal (~0.1 ms)
          </div>
        </div>
      ) : (
        /* Quantum Circuit SVG Visualizer */
        <div className="overflow-x-auto bg-slate-950/80 border border-slate-800 rounded-xl p-6 my-2">
          <svg className="w-full min-w-[700px] h-[260px]" viewBox="0 0 820 250">
            {/* Qubit Wire Lines */}
            {qubits.map((q) => {
              const y = 40 + q * 52;
              return (
                <g key={q}>
                  {/* Qubit Label */}
                  <text x="20" y={y + 5} fill="#94a3b8" fontSize="12" fontFamily="monospace" fontWeight="bold">
                    |q{q}⟩
                  </text>
                  <text x="50" y={y + 5} fill="#64748b" fontSize="11" fontFamily="monospace">
                    |0⟩
                  </text>

                  {/* Wire line */}
                  <line x1="75" y1={y} x2={activePathView === 1 ? "520" : "740"} y2={y} stroke="#334155" strokeWidth="2" />

                  {/* RY Angle Embedding Gate */}
                  <rect x="110" y={y - 18} width="65" height="36" rx="6" fill="#1e1b4b" stroke="#6366f1" strokeWidth="1.5" />
                  <text x="142" y={y + 4} fill="#c7d2fe" fontSize="10" fontFamily="monospace" textAnchor="middle" fontWeight="bold">
                    RY(u{q})
                  </text>

                  {/* Layer 1: Rot Gate */}
                  <rect x="205" y={y - 18} width="65" height="36" rx="6" fill="#451a03" stroke="#f59e0b" strokeWidth="1.5" />
                  <text x="237" y={y + 4} fill="#fef3c7" fontSize="10" fontFamily="monospace" textAnchor="middle" fontWeight="bold">
                    Rot(θ₁,{q})
                  </text>

                  {/* Layer 2 (if Path 2): Rot Gate */}
                  {activePathView === 2 && (
                    <>
                      <rect x="420" y={y - 18} width="65" height="36" rx="6" fill="#3b0764" stroke="#a855f7" strokeWidth="1.5" />
                      <text x="452" y={y + 4} fill="#f3e8ff" fontSize="10" fontFamily="monospace" textAnchor="middle" fontWeight="bold">
                        Rot(θ₂,{q})
                      </text>
                    </>
                  )}

                  {/* Pauli-Z Measurement Box */}
                  <rect 
                    x={activePathView === 1 ? "440" : "650"} 
                    y={y - 18} 
                    width="55" 
                    height="36" 
                    rx="6" 
                    fill="#0f172a" 
                    stroke="#38bdf8" 
                    strokeWidth="1.5" 
                  />
                  <text 
                    x={activePathView === 1 ? "467" : "677"} 
                    y={y + 4} 
                    fill="#38bdf8" 
                    fontSize="11" 
                    fontFamily="monospace" 
                    textAnchor="middle" 
                    fontWeight="bold"
                  >
                    ⟨Z{q}⟩
                  </text>
                </g>
              );
            })}

            {/* Circular CNOT Entanglers Layer 1 (x: 295 - 390) */}
            {/* CNOT 0 -> 1 */}
            <circle cx="300" cy="40" r="4" fill="#38bdf8" />
            <line x1="300" y1="40" x2="300" y2="92" stroke="#38bdf8" strokeWidth="1.5" />
            <circle cx="300" cy="92" r="7" fill="none" stroke="#38bdf8" strokeWidth="1.5" />
            <line x1="293" y1="92" x2="307" y2="92" stroke="#38bdf8" strokeWidth="1.5" />
            <line x1="300" y1="85" x2="300" y2="99" stroke="#38bdf8" strokeWidth="1.5" />

            {/* CNOT 1 -> 2 */}
            <circle cx="330" cy="92" r="4" fill="#38bdf8" />
            <line x1="330" y1="92" x2="330" y2="144" stroke="#38bdf8" strokeWidth="1.5" />
            <circle cx="330" cy="144" r="7" fill="none" stroke="#38bdf8" strokeWidth="1.5" />
            <line x1="323" y1="144" x2="337" y2="144" stroke="#38bdf8" strokeWidth="1.5" />
            <line x1="330" y1="137" x2="330" y2="151" stroke="#38bdf8" strokeWidth="1.5" />

            {/* CNOT 2 -> 3 */}
            <circle cx="360" cy="144" r="4" fill="#38bdf8" />
            <line x1="360" y1="144" x2="360" y2="196" stroke="#38bdf8" strokeWidth="1.5" />
            <circle cx="360" cy="196" r="7" fill="none" stroke="#38bdf8" strokeWidth="1.5" />
            <line x1="353" y1="196" x2="367" y2="196" stroke="#38bdf8" strokeWidth="1.5" />
            <line x1="360" y1="189" x2="360" y2="203" stroke="#38bdf8" strokeWidth="1.5" />

            {/* CNOT 3 -> 0 (Circular Ring) */}
            <circle cx="390" cy="196" r="4" fill="#38bdf8" />
            <line x1="390" y1="196" x2="390" y2="40" stroke="#38bdf8" strokeWidth="1.5" strokeDasharray="3,3" />
            <circle cx="390" cy="40" r="7" fill="none" stroke="#38bdf8" strokeWidth="1.5" />
            <line x1="383" y1="40" x2="397" y2="40" stroke="#38bdf8" strokeWidth="1.5" />
            <line x1="390" y1="33" x2="390" y2="47" stroke="#38bdf8" strokeWidth="1.5" />

            {/* Circular CNOT Entanglers Layer 2 (if Path 2, x: 515 - 610) */}
            {activePathView === 2 && (
              <>
                {/* CNOT 0 -> 1 */}
                <circle cx="515" cy="40" r="4" fill="#c084fc" />
                <line x1="515" y1="40" x2="515" y2="92" stroke="#c084fc" strokeWidth="1.5" />
                <circle cx="515" cy="92" r="7" fill="none" stroke="#c084fc" strokeWidth="1.5" />
                <line x1="508" y1="92" x2="522" y2="92" stroke="#c084fc" strokeWidth="1.5" />
                <line x1="515" y1="85" x2="515" y2="99" stroke="#c084fc" strokeWidth="1.5" />

                {/* CNOT 1 -> 2 */}
                <circle cx="545" cy="92" r="4" fill="#c084fc" />
                <line x1="545" y1="92" x2="545" y2="144" stroke="#c084fc" strokeWidth="1.5" />
                <circle cx="545" cy="144" r="7" fill="none" stroke="#c084fc" strokeWidth="1.5" />
                <line x1="538" y1="144" x2="552" y2="144" stroke="#c084fc" strokeWidth="1.5" />
                <line x1="545" y1="137" x2="545" y2="151" stroke="#c084fc" strokeWidth="1.5" />

                {/* CNOT 2 -> 3 */}
                <circle cx="575" cy="144" r="4" fill="#c084fc" />
                <line x1="575" y1="144" x2="575" y2="196" stroke="#c084fc" strokeWidth="1.5" />
                <circle cx="575" cy="196" r="7" fill="none" stroke="#c084fc" strokeWidth="1.5" />
                <line x1="568" y1="196" x2="582" y2="196" stroke="#c084fc" strokeWidth="1.5" />
                <line x1="575" y1="189" x2="575" y2="203" stroke="#c084fc" strokeWidth="1.5" />

                {/* CNOT 3 -> 0 (Circular Ring) */}
                <circle cx="605" cy="196" r="4" fill="#c084fc" />
                <line x1="605" y1="196" x2="605" y2="40" stroke="#c084fc" strokeWidth="1.5" strokeDasharray="3,3" />
                <circle cx="605" cy="40" r="7" fill="none" stroke="#c084fc" strokeWidth="1.5" />
                <line x1="598" y1="40" x2="612" y2="40" stroke="#c084fc" strokeWidth="1.5" />
                <line x1="605" y1="33" x2="605" y2="47" stroke="#c084fc" strokeWidth="1.5" />
              </>
            )}

            {/* Column Header Annotations */}
            <text x="142" y="15" fill="#818cf8" fontSize="10" fontFamily="monospace" textAnchor="middle">
              State Preparation
            </text>
            <text x="345" y="15" fill="#fbbf24" fontSize="10" fontFamily="monospace" textAnchor="middle">
              Variational Layer 1
            </text>
            {activePathView === 2 && (
              <text x="560" y="15" fill="#c084fc" fontSize="10" fontFamily="monospace" textAnchor="middle">
                Variational Layer 2
              </text>
            )}
            <text 
              x={activePathView === 1 ? "467" : "677"} 
              y="15" 
              fill="#38bdf8" 
              fontSize="10" 
              fontFamily="monospace" 
              textAnchor="middle"
            >
              Pauli-Z
            </text>
          </svg>
        </div>
      )}

      {/* Expectation Values & Latent Vectors if active */}
      {activePathView > 0 && pauliExpectations && (
        <div className="mt-4 p-4 rounded-xl bg-slate-900/90 border border-slate-800">
          <div className="text-xs font-mono text-slate-300 font-bold mb-2 flex items-center gap-1.5">
            <Sparkles className="w-4 h-4 text-brand-400" />
            Measured Pauli-Z Expectation Values ⟨Zⱼ⟩ ∈ [-1.0, 1.0]:
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {pauliExpectations.map((expVal, idx) => (
              <div key={idx} className="bg-slate-950 p-2 rounded-lg border border-slate-800 font-mono text-center">
                <span className="text-[10px] text-slate-400 block">Qubit {idx} ⟨Z_{idx}⟩</span>
                <span className="text-xs font-bold text-brand-300">{expVal.toFixed(4)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
