import React, { useState, useEffect } from 'react';
import { getDatasets } from '../services/api';
import { DatasetMetadata } from '../types';
import { Database, ShieldCheck, FileSpreadsheet, CheckCircle2, Sliders, Info } from 'lucide-react';
import { Link } from 'react-router-dom';

export const DatasetsPage: React.FC = () => {
  const [datasets, setDatasets] = useState<DatasetMetadata[]>([]);
  const [selectedDatasetKey, setSelectedDatasetKey] = useState<string>('breast_cancer');

  useEffect(() => {
    const fetchDs = async () => {
      try {
        const data = await getDatasets();
        setDatasets(data);
      } catch (e) {
        console.error(e);
      }
    };
    fetchDs();
  }, []);

  const currentDs = datasets.find((d) => d.key === selectedDatasetKey);

  return (
    <div className="space-y-8 py-4 max-w-5xl mx-auto">
      <div>
        <h1 className="text-2xl font-black text-white tracking-tight flex items-center gap-2">
          <Database className="w-6 h-6 text-brand-400" />
          Clinical Benchmark Datasets & Preprocessing
        </h1>
        <p className="text-xs text-slate-400 mt-1">
          Detailed dataset characteristics, class balance distributions, and leakage-safe preprocessing pipelines
        </p>
      </div>

      {/* Dataset Selector Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {datasets.map((ds) => (
          <button
            key={ds.key}
            onClick={() => setSelectedDatasetKey(ds.key)}
            className={`p-5 rounded-2xl text-left border transition-all ${
              selectedDatasetKey === ds.key
                ? 'bg-brand-500/20 border-brand-400 shadow-lg shadow-brand-500/10'
                : 'glass-panel border-slate-800 hover:border-slate-700'
            }`}
          >
            <div className="text-xs font-mono text-brand-400 uppercase tracking-wider">{ds.domain}</div>
            <h3 className="text-base font-bold text-white mt-1">{ds.display_name}</h3>
            <div className="grid grid-cols-2 gap-2 mt-4 text-xs font-mono text-slate-300">
              <div className="bg-slate-950/80 p-2 rounded border border-slate-800">
                <span className="text-[10px] text-slate-400 block">Samples (N)</span>
                <span className="font-bold text-white">{ds.num_samples}</span>
              </div>
              <div className="bg-slate-950/80 p-2 rounded border border-slate-800">
                <span className="text-[10px] text-slate-400 block">Features (D)</span>
                <span className="font-bold text-white">{ds.num_features}</span>
              </div>
            </div>
          </button>
        ))}
      </div>

      {/* Selected Dataset Detail Panel */}
      {currentDs && (
        <div className="glass-panel rounded-2xl p-6 border border-slate-800 shadow-xl space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
            <div>
              <span className="text-[11px] font-mono text-brand-400 uppercase">{currentDs.domain}</span>
              <h2 className="text-xl font-bold text-white mt-0.5">{currentDs.display_name}</h2>
            </div>
            <Link
              to="/diagnosis"
              className="px-4 py-2 rounded-xl bg-brand-600 hover:bg-brand-500 text-xs font-bold text-white flex items-center gap-1.5 w-fit"
            >
              <Sliders className="w-4 h-4" />
              Test on Diagnosis Page
            </Link>
          </div>

          {/* Class Distribution & Missingness Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="bg-slate-900/80 p-4 rounded-xl border border-slate-800">
              <span className="text-xs text-slate-400 font-mono block">Class 0 ({currentDs.target_negative})</span>
              <span className="text-2xl font-bold font-mono text-emerald-400 mt-1 block">
                {currentDs.class_0_count} <span className="text-xs text-slate-400 font-normal">({((currentDs.class_0_count/currentDs.num_samples)*100).toFixed(1)}%)</span>
              </span>
            </div>

            <div className="bg-slate-900/80 p-4 rounded-xl border border-slate-800">
              <span className="text-xs text-slate-400 font-mono block">Class 1 ({currentDs.target_positive})</span>
              <span className="text-2xl font-bold font-mono text-rose-400 mt-1 block">
                {currentDs.class_1_count} <span className="text-xs text-slate-400 font-normal">({((currentDs.class_1_count/currentDs.num_samples)*100).toFixed(1)}%)</span>
              </span>
            </div>

            <div className="bg-slate-900/80 p-4 rounded-xl border border-slate-800">
              <span className="text-xs text-slate-400 font-mono block">Missing Values</span>
              <span className="text-2xl font-bold font-mono text-brand-300 mt-1 block">
                {currentDs.missing_pct}%
              </span>
              <span className="text-[10px] text-slate-400 block mt-0.5">Median Imputed (Train Only)</span>
            </div>
          </div>

          {/* Preprocessing Protocol */}
          <div className="bg-slate-950/80 p-4 rounded-xl border border-slate-800 space-y-2">
            <h4 className="text-xs font-bold text-white font-mono uppercase tracking-wider flex items-center gap-1.5">
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
              Leakage-Safe Preprocessing Protocol
            </h4>
            <p className="text-xs text-slate-300 leading-relaxed">
              All statistical transformations (median imputers, standard scalers $\mu, \sigma$) are fitted <strong>strictly on outer training folds</strong>. The untouched outer validation fold and isolated holdout test set are transformed using frozen training statistics to guarantee zero data leakage.
            </p>
          </div>

          {/* Feature List Table */}
          <div className="space-y-3">
            <h4 className="text-xs font-bold text-white font-mono uppercase tracking-wider">
              Feature Metadata & Bounds ({currentDs.features.length} Features)
            </h4>
            <div className="overflow-x-auto max-h-72 overflow-y-auto">
              <table className="w-full text-left text-xs border-collapse">
                <thead className="sticky top-0 bg-slate-900 font-mono text-slate-400 text-[11px]">
                  <tr className="border-b border-slate-800">
                    <th className="py-2 px-3">Feature Name</th>
                    <th className="py-2 px-3">Description</th>
                    <th className="py-2 px-3">Unit</th>
                    <th className="py-2 px-3">Range [Min - Max]</th>
                    <th className="py-2 px-3">Default</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono">
                  {currentDs.features.map((f) => (
                    <tr key={f.name} className="hover:bg-slate-900/40">
                      <td className="py-2 px-3 text-brand-300 font-semibold">{f.display_name}</td>
                      <td className="py-2 px-3 text-slate-300 font-sans text-[11px]">{f.description}</td>
                      <td className="py-2 px-3 text-slate-400">{f.unit || '—'}</td>
                      <td className="py-2 px-3 text-slate-300">[{f.min_value} - {f.max_value}]</td>
                      <td className="py-2 px-3 text-emerald-400">{f.default_value}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

        </div>
      )}
    </div>
  );
};
