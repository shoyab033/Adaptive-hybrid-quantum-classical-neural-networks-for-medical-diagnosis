import React, { useState } from 'react';
import { QuantumCircuitViewer } from '../components/diagnosis/QuantumCircuitViewer';
import { Cpu, Info, Layers, Zap, Sparkles } from 'lucide-react';

export const QuantumCircuitPage: React.FC = () => {
  const [selectedDemoPath, setSelectedDemoPath] = useState<number>(1);

  return (
    <div className="space-y-8 py-4 max-w-5xl mx-auto">
      <div>
        <h1 className="text-2xl font-black text-white tracking-tight flex items-center gap-2">
          <Cpu className="w-6 h-6 text-quantum-400" />
          Quantum Circuit Architecture & Simulation Details
        </h1>
        <p className="text-xs text-slate-400 mt-1">
          Interactive quantum gate-level layout, parameterized Euler rotations, and state-vector simulation specifics
        </p>
      </div>

      {/* Main Interactive Circuit Viewer */}
      <QuantumCircuitViewer selectedPath={selectedDemoPath} />

      {/* Circuit Specifications Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="glass-panel rounded-2xl p-6 border border-slate-800 space-y-3">
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <Layers className="w-4 h-4 text-brand-400" />
            Gate Formulation & Parameters
          </h3>
          <ul className="space-y-2 text-xs text-slate-300">
            <li className="flex items-start gap-2">
              <span className="text-brand-400 font-mono font-bold">•</span>
              <span><strong>Number of Qubits:</strong> 4 Qubits ($q_0, q_1, q_2, q_3$) simulated via PennyLane `default.qubit`.</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-brand-400 font-mono font-bold">•</span>
              <span><strong>State Preparation:</strong> $R_Y(u_j)$ Angle Embedding mapping latent angles $u \in [-\pi, \pi]^4$.</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-brand-400 font-mono font-bold">•</span>
              <span><strong>Single-Qubit Rotations:</strong> Arbitrary Euler rotation $R(\alpha, \beta, \gamma) = R_Z(\gamma) R_Y(\beta) R_Z(\alpha)$ with 3 trainable parameters per qubit (12 parameters/layer).</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-brand-400 font-mono font-bold">•</span>
              <span><strong>Entanglement Topology:</strong> Circular ring CNOT gates connecting $(0\to 1), (1\to 2), (2\to 3), (3\to 0)$.</span>
            </li>
          </ul>
        </div>

        <div className="glass-panel rounded-2xl p-6 border border-slate-800 space-y-3">
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <Zap className="w-4 h-4 text-amber-400" />
            Dynamic Depth Selection
          </h3>
          <ul className="space-y-2 text-xs text-slate-300">
            <li className="flex items-start gap-2">
              <span className="text-emerald-400 font-mono font-bold">•</span>
              <span><strong>Path 0:</strong> 0 Quantum Layers. Circuit execution is completely bypassed when classical prediction is decisive.</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-amber-400 font-mono font-bold">•</span>
              <span><strong>Path 1:</strong> 1 Quantum Layer. Evaluates 4 Rot gates + 4 CNOTs for intermediate uncertainty instances.</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-purple-400 font-mono font-bold">•</span>
              <span><strong>Path 2:</strong> 2 Quantum Layers. Evaluates 8 Rot gates + 8 CNOTs for high-uncertainty borderline instances.</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-brand-400 font-mono font-bold">•</span>
              <span><strong>Readout:</strong> Expectation values of Pauli-$Z$ observable $\langle Z_j \rangle \in [-1, 1]^4$ mapped via linear heads.</span>
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
};
