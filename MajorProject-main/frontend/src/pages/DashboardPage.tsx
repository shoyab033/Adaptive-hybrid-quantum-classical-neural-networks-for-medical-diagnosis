import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { 
  getPerformance, 
  getQuantumWorkload, 
  getDatasets 
} from '../services/api';
import { 
  ModelPerformanceItem, 
  QuantumWorkloadItem, 
  DatasetMetadata 
} from '../types';
import { 
  BarChart3, 
  Cpu, 
  Activity, 
  ShieldCheck, 
  Zap, 
  Layers, 
  TrendingDown,
  ArrowRight
} from 'lucide-react';
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

export const DashboardPage: React.FC = () => {
  const [performance, setPerformance] = useState<ModelPerformanceItem[]>([]);
  const [workload, setWorkload] = useState<QuantumWorkloadItem[]>([]);
  const [datasets, setDatasets] = useState<DatasetMetadata[]>([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [perfData, workData, dsData] = await Promise.all([
          getPerformance(),
          getQuantumWorkload(),
          getDatasets(),
        ]);
        setPerformance(perfData);
        setWorkload(workData);
        setDatasets(dsData);
      } catch (e) {
        console.error('Error loading dashboard data', e);
      }
    };
    fetchData();
  }, []);

  const workloadChartData = workload.map((w) => ({
    name: w.dataset.replace(' Breast Cancer', '').replace('UCI ', '').replace(' Indians Database', ''),
    'Path 0 (0 QL)': w.classical_only_pct,
    'Path 1 (1 QL)': w.one_qlayer_pct,
    'Path 2 (2 QL)': w.two_qlayers_pct,
  }));

  const layersChartData = workload.map((w) => ({
    name: w.dataset.replace(' Breast Cancer', '').replace('UCI ', '').replace(' Indians Database', ''),
    'Fixed HQNN (M2)': 2.0,
    'Adaptive HQNN (M3)': w.avg_quantum_layers,
  }));

  return (
    <div className="space-y-8 py-4">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-black text-white tracking-tight flex items-center gap-2">
            <BarChart3 className="w-6 h-6 text-brand-400" />
            Research Overview Dashboard
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Empirical metrics, inference routing distributions, and quantum workload reduction across all 3 medical benchmarks
          </p>
        </div>

        <Link
          to="/diagnosis"
          className="self-start sm:self-auto px-4 py-2 rounded-xl bg-brand-600 hover:bg-brand-500 text-xs font-bold text-white shadow-lg shadow-brand-600/30 transition-all flex items-center gap-1.5"
        >
          <Activity className="w-4 h-4" />
          Test Live Model
        </Link>
      </div>

      {/* KPI Cards Strip */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="glass-panel rounded-2xl p-5 border border-slate-800">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-mono text-slate-400 uppercase">WDBC Q-Reduction</span>
            <TrendingDown className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-3xl font-black font-mono text-emerald-400 mt-2">98.8%</div>
          <p className="text-[11px] text-slate-400 mt-1">0.02 vs 2.00 Q-Layers per sample</p>
        </div>

        <div className="glass-panel rounded-2xl p-5 border border-slate-800">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-mono text-slate-400 uppercase">Heart Q-Reduction</span>
            <TrendingDown className="w-4 h-4 text-brand-400" />
          </div>
          <div className="text-3xl font-black font-mono text-brand-400 mt-2">87.8%</div>
          <p className="text-[11px] text-slate-400 mt-1">0.24 vs 2.00 Q-Layers per sample</p>
        </div>

        <div className="glass-panel rounded-2xl p-5 border border-slate-800">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-mono text-slate-400 uppercase">Diabetes Q-Reduction</span>
            <TrendingDown className="w-4 h-4 text-purple-400" />
          </div>
          <div className="text-3xl font-black font-mono text-purple-400 mt-2">79.6%</div>
          <p className="text-[11px] text-slate-400 mt-1">0.41 vs 2.00 Q-Layers per sample</p>
        </div>

        <div className="glass-panel rounded-2xl p-5 border border-slate-800">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-mono text-slate-400 uppercase">Statistical Rigor</span>
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-xl font-bold font-mono text-white mt-2">p ≥ 0.05</div>
          <p className="text-[11px] text-slate-400 mt-1">Paired Wilcoxon test (No sig. loss in AUC)</p>
        </div>
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        
        {/* Stacked Routing Distribution Chart */}
        <div className="glass-panel rounded-2xl p-6 border border-slate-800 shadow-xl space-y-4">
          <div>
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <Zap className="w-4 h-4 text-amber-400" />
              Inference Routing Pathway Allocation (%)
            </h3>
            <p className="text-[11px] text-slate-400 mt-0.5">
              Proportion of patient records resolved via Path 0 (0 QL), Path 1 (1 QL), and Path 2 (2 QL)
            </p>
          </div>

          <div className="h-64 w-full pt-4">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={workloadChartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
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

        {/* Avg Quantum Layers Comparison Chart */}
        <div className="glass-panel rounded-2xl p-6 border border-slate-800 shadow-xl space-y-4">
          <div>
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <Cpu className="w-4 h-4 text-purple-400" />
              Average Inference Quantum Layers per Sample
            </h3>
            <p className="text-[11px] text-slate-400 mt-0.5">
              Comparison between Fixed-Depth HQNN ($L_Q = 2.0$) and Proposed Adaptive HQNN
            </p>
          </div>

          <div className="h-64 w-full pt-4">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={layersChartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="name" stroke="#94a3b8" fontSize={11} />
                <YAxis stroke="#94a3b8" fontSize={11} />
                <Tooltip contentStyle={{ backgroundColor: '#090d16', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }} />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
                <Bar dataKey="Fixed HQNN (M2)" fill="#475569" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Adaptive HQNN (M3)" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

      </div>

      {/* Cross-Validation Performance Preview Table */}
      <div className="glass-panel rounded-2xl p-6 border border-slate-800 shadow-xl space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <Layers className="w-4 h-4 text-brand-400" />
              Empirical Performance Summary (5-Fold Cross-Validation)
            </h3>
            <p className="text-[11px] text-slate-400 mt-0.5">
              Authoritative results across all 4 benchmark models
            </p>
          </div>
          <Link to="/performance" className="text-xs text-brand-400 hover:text-brand-300 font-semibold flex items-center gap-1">
            Full Table <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-800 bg-slate-900/60 font-mono text-slate-400 text-[11px]">
                <th className="py-2.5 px-3">Dataset</th>
                <th className="py-2.5 px-3">Model</th>
                <th className="py-2.5 px-3">CV Accuracy</th>
                <th className="py-2.5 px-3">CV F1-Score</th>
                <th className="py-2.5 px-3">CV ROC-AUC</th>
                <th className="py-2.5 px-3">Holdout Test F1</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {performance.slice(0, 8).map((p, idx) => (
                <tr key={idx} className={p.model_id === 'M3' ? 'bg-brand-500/10 font-semibold' : 'hover:bg-slate-900/40'}>
                  <td className="py-2 px-3 text-slate-300">{p.dataset}</td>
                  <td className="py-2 px-3 text-white flex items-center gap-1.5">
                    <span className="font-mono text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-brand-300">{p.model_id}</span>
                    {p.model}
                  </td>
                  <td className="py-2 px-3 font-mono text-slate-300">{p.cv_accuracy}</td>
                  <td className="py-2 px-3 font-mono text-slate-300">{p.cv_f1}</td>
                  <td className="py-2 px-3 font-mono text-brand-300">{p.cv_roc_auc}</td>
                  <td className="py-2 px-3 font-mono text-emerald-400">{p.test_f1}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
