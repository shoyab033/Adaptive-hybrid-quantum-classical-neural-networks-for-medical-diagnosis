import React, { useState, useEffect } from 'react';
import { getQuantumWorkload } from '../services/api';
import { QuantumWorkloadItem } from '../types';
import { Cpu, Zap, TrendingDown, Info, ShieldAlert } from 'lucide-react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  Tooltip, 
  Legend, 
  ResponsiveContainer, 
  CartesianGrid 
} from 'recharts';

export const QuantumWorkloadPage: React.FC = () => {
  const [workload, setWorkload] = useState<QuantumWorkloadItem[]>([]);

  useEffect(() => {
    const fetchWorkload = async () => {
      try {
        const data = await getQuantumWorkload();
        setWorkload(data);
      } catch (e) {
        console.error(e);
      }
    };
    fetchWorkload();
  }, []);

  const chartData = workload.map((w) => ({
    name: w.dataset.replace(' Breast Cancer', '').replace('UCI ', '').replace(' Indians Database', ''),
    'Path 0 (0 QL)': w.classical_only_pct,
    'Path 1 (1 QL)': w.one_qlayer_pct,
    'Path 2 (2 QL)': w.two_qlayers_pct,
  }));

  return (
    <div className="space-y-8 py-4 max-w-5xl mx-auto">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-black text-white tracking-tight flex items-center gap-2">
          <Cpu className="w-6 h-6 text-quantum-400" />
          Quantum Workload & Inference Efficiency (Table III)
        </h1>
        <p className="text-xs text-slate-400 mt-1">
          Empirical evaluation of estimated inference-time quantum layer evaluations and dynamic routing allocations
        </p>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {workload.map((w) => (
          <div key={w.dataset} className="glass-panel rounded-2xl p-6 border border-slate-800 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono text-brand-400 uppercase tracking-wider">{w.dataset}</span>
              <TrendingDown className="w-4 h-4 text-emerald-400" />
            </div>
            <div>
              <div className="text-4xl font-black font-mono text-emerald-400">
                {w.workload_reduction_pct}%
              </div>
              <span className="text-xs font-semibold text-slate-200 block mt-0.5">Estimated Quantum Workload Saved</span>
            </div>
            <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800 space-y-1.5 text-xs font-mono text-slate-300">
              <div className="flex justify-between">
                <span className="text-slate-400">Avg Q-Layers / Sample:</span>
                <span className="font-bold text-white">{w.avg_quantum_layers}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Path 0 (0 QL):</span>
                <span className="text-emerald-400 font-bold">{w.classical_only_pct}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Path 1 (1 QL):</span>
                <span className="text-amber-400 font-bold">{w.one_qlayer_pct}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Path 2 (2 QL):</span>
                <span className="text-purple-400 font-bold">{w.two_qlayers_pct}%</span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Stacked Chart */}
      <div className="glass-panel rounded-2xl p-6 border border-slate-800 shadow-xl space-y-4">
        <div>
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <Zap className="w-4 h-4 text-amber-400" />
            Routing Pathway Distribution by Disease Modality (%)
          </h3>
          <p className="text-[11px] text-slate-400 mt-0.5">
            How sample complexity directly governs quantum circuit depth allocation
          </p>
        </div>

        <div className="h-72 w-full pt-4">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 10, right: 20, left: -20, bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="name" stroke="#94a3b8" fontSize={11} />
              <YAxis stroke="#94a3b8" fontSize={11} unit="%" />
              <Tooltip contentStyle={{ backgroundColor: '#090d16', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }} />
              <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
              <Bar dataKey="Path 0 (0 QL)" stackId="a" fill="#10b981" radius={[0, 0, 4, 4]} />
              <Bar dataKey="Path 1 (1 QL)" stackId="a" fill="#f59e0b" />
              <Bar dataKey="Path 2 (2 QL)" stackId="a" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Scope and Scientific Note */}
      <div className="bg-slate-950/80 p-5 rounded-2xl border border-slate-800 space-y-2">
        <h4 className="text-xs font-bold text-white font-mono uppercase tracking-wider flex items-center gap-1.5">
          <ShieldAlert className="w-4 h-4 text-amber-400" />
          Quantum Workload Metric Scope & Transparency
        </h4>
        <p className="text-xs text-slate-300 leading-relaxed">
          The term <strong>"estimated inference-time quantum workload reduction"</strong> refers explicitly to the percentage reduction in the expected number of variational quantum layer executions (E[L_Q(x)]) evaluated per patient instance relative to the static depth-2 HQNN baseline. It does not represent direct measurements of physical quantum processor cryogenic power or compiler gate compilation durations.
        </p>
      </div>

    </div>
  );
};
