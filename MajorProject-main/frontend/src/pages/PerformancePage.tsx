import React, { useState, useEffect } from 'react';
import { getPerformance } from '../services/api';
import { ModelPerformanceItem } from '../types';
import { Sliders, Filter, ShieldCheck, CheckCircle2, FileSpreadsheet, ArrowRight } from 'lucide-react';
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

export const PerformancePage: React.FC = () => {
  const [performance, setPerformance] = useState<ModelPerformanceItem[]>([]);
  const [selectedDatasetFilter, setSelectedDatasetFilter] = useState<string>('all');

  useEffect(() => {
    const fetchPerf = async () => {
      try {
        const data = await getPerformance();
        setPerformance(data);
      } catch (e) {
        console.error(e);
      }
    };
    fetchPerf();
  }, []);

  const filtered = selectedDatasetFilter === 'all'
    ? performance
    : performance.filter((p) => p.dataset.toLowerCase().includes(selectedDatasetFilter.toLowerCase()));

  // Chart data: Test F1 and ROC-AUC for filtered models
  const chartData = filtered.map((p) => ({
    name: `${p.model_id} (${p.dataset.split(' ')[0]})`,
    'Test F1 (%)': parseFloat(p.test_f1.replace('%', '')),
    'Test ROC-AUC': parseFloat(p.test_roc_auc) * 100,
  }));

  return (
    <div className="space-y-8 py-4 max-w-5xl mx-auto">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-black text-white tracking-tight flex items-center gap-2">
            <Sliders className="w-6 h-6 text-brand-400" />
            Model Performance Benchmarks (Table II)
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Authoritative empirical results comparing Classical MLP (M1), Fixed HQNN (M2), Adaptive HQNN (M3), and Adaptive Classical (M4)
          </p>
        </div>

        {/* Filter Pills */}
        <div className="flex items-center gap-1.5 bg-slate-900 border border-slate-800 p-1.5 rounded-xl">
          <button
            onClick={() => setSelectedDatasetFilter('all')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              selectedDatasetFilter === 'all'
                ? 'bg-brand-600 text-white'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            All Datasets
          </button>
          <button
            onClick={() => setSelectedDatasetFilter('wdbc')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              selectedDatasetFilter === 'wdbc'
                ? 'bg-brand-600 text-white'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            WDBC
          </button>
          <button
            onClick={() => setSelectedDatasetFilter('heart')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              selectedDatasetFilter === 'heart'
                ? 'bg-brand-600 text-white'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            Heart
          </button>
          <button
            onClick={() => setSelectedDatasetFilter('diabetes')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              selectedDatasetFilter === 'diabetes'
                ? 'bg-brand-600 text-white'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            Diabetes
          </button>
        </div>
      </div>

      {/* Comparison Chart */}
      <div className="glass-panel rounded-2xl p-6 border border-slate-800 shadow-xl space-y-4">
        <h3 className="text-sm font-bold text-white font-mono uppercase tracking-wider">
          Holdout Test Evaluation Metrics (F1-Score & ROC-AUC)
        </h3>
        <div className="h-72 w-full pt-2">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 10, right: 20, left: -20, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="name" stroke="#94a3b8" fontSize={10} angle={-15} textAnchor="end" />
              <YAxis stroke="#94a3b8" fontSize={11} domain={[50, 100]} />
              <Tooltip contentStyle={{ backgroundColor: '#090d16', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }} />
              <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
              <Bar dataKey="Test F1 (%)" fill="#0ea5e9" radius={[4, 4, 0, 0]} />
              <Bar dataKey="Test ROC-AUC" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Full Benchmark Table */}
      <div className="glass-panel rounded-2xl p-6 border border-slate-800 shadow-xl space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <FileSpreadsheet className="w-4 h-4 text-brand-400" />
            Complete Empirical Cross-Validation and Holdout Test Matrix
          </h3>
          <span className="text-[11px] font-mono text-slate-400">5-Fold Stratified CV</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-800 bg-slate-900/60 font-mono text-slate-400 text-[11px]">
                <th className="py-3 px-3">Dataset</th>
                <th className="py-3 px-3">Model Architecture</th>
                <th className="py-3 px-3">CV Accuracy</th>
                <th className="py-3 px-3">CV F1-Score</th>
                <th className="py-3 px-3">CV ROC-AUC</th>
                <th className="py-3 px-3">Test F1</th>
                <th className="py-3 px-3">Test ROC-AUC</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {filtered.map((p, idx) => {
                const isProposed = p.model_id === 'M3';
                return (
                  <tr key={idx} className={isProposed ? 'bg-brand-500/15 font-bold text-white' : 'hover:bg-slate-900/40 text-slate-300'}>
                    <td className="py-2.5 px-3">{p.dataset}</td>
                    <td className="py-2.5 px-3 flex items-center gap-1.5 font-sans">
                      <span className={`px-1.5 py-0.5 rounded text-[10px] font-mono ${
                        isProposed ? 'bg-brand-500 text-slate-950 font-black' : 'bg-slate-800 text-slate-300'
                      }`}>
                        {p.model_id}
                      </span>
                      {p.model}
                    </td>
                    <td className="py-2.5 px-3">{p.cv_accuracy}</td>
                    <td className="py-2.5 px-3">{p.cv_f1}</td>
                    <td className="py-2.5 px-3 text-brand-300">{p.cv_roc_auc}</td>
                    <td className="py-2.5 px-3 text-emerald-400">{p.test_f1}</td>
                    <td className="py-2.5 px-3 text-purple-400">{p.test_roc_auc}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Statistical Rigor Card */}
      <div className="bg-slate-950/80 p-5 rounded-2xl border border-slate-800 space-y-2">
        <h4 className="text-xs font-bold text-white font-mono uppercase tracking-wider flex items-center gap-1.5">
          <ShieldCheck className="w-4 h-4 text-emerald-400" />
          Paired Wilcoxon Signed-Rank Hypothesis Testing
        </h4>
        <p className="text-xs text-slate-300 leading-relaxed">
          Across all three clinical datasets, paired Wilcoxon signed-rank tests between Adaptive HQNN (M3) and Fixed HQNN (M2) yield $p \ge 0.05$ ($p=0.125$ on WDBC, $p=0.0625$ on Heart, $p=0.0625$ on Diabetes). Under standard statistical protocols, no statistically significant difference in ROC-AUC was detected while achieving up to 98.8% estimated quantum layer reduction.
        </p>
      </div>

    </div>
  );
};
