import React, { useState, useEffect } from 'react';
import { PredictionHistoryRecord } from '../types';
import { History, Trash2, Download, Search, Zap, Clock, Activity } from 'lucide-react';
import { Link } from 'react-router-dom';

export const PredictionHistoryPage: React.FC = () => {
  const [history, setHistory] = useState<PredictionHistoryRecord[]>([]);
  const [search, setSearch] = useState('');

  useEffect(() => {
    const saved = localStorage.getItem('adaptive_hqnn_history');
    if (saved) {
      try {
        setHistory(JSON.parse(saved));
      } catch (e) {
        console.error(e);
      }
    }
  }, []);

  const handleClearHistory = () => {
    if (window.confirm('Are you sure you want to clear your local prediction history?')) {
      localStorage.removeItem('adaptive_hqnn_history');
      setHistory([]);
    }
  };

  const handleExportJSON = () => {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(history, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", "adaptive_hqnn_prediction_history.json");
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  const filteredHistory = history.filter((h) =>
    h.dataset_name.toLowerCase().includes(search.toLowerCase()) ||
    h.prediction_label.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="space-y-8 py-4 max-w-5xl mx-auto">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-black text-white tracking-tight flex items-center gap-2">
            <History className="w-6 h-6 text-brand-400" />
            Prediction & Routing History Log
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Locally stored patient predictions, confidence scores, and quantum routing path assignments
          </p>
        </div>

        <div className="flex items-center gap-2">
          {history.length > 0 && (
            <>
              <button
                onClick={handleExportJSON}
                className="px-3.5 py-1.5 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-700 text-xs font-medium text-slate-200 transition-colors flex items-center gap-1.5"
              >
                <Download className="w-3.5 h-3.5" />
                Export JSON
              </button>
              <button
                onClick={handleClearHistory}
                className="px-3.5 py-1.5 rounded-xl bg-rose-500/10 hover:bg-rose-500/20 border border-rose-500/30 text-xs font-medium text-rose-300 transition-colors flex items-center gap-1.5"
              >
                <Trash2 className="w-3.5 h-3.5" />
                Clear
              </button>
            </>
          )}
        </div>
      </div>

      {history.length === 0 ? (
        <div className="glass-panel rounded-2xl p-12 text-center border border-slate-800 space-y-4">
          <Clock className="w-10 h-10 text-slate-600 mx-auto" />
          <h3 className="text-sm font-bold text-white">No Prediction History Found</h3>
          <p className="text-xs text-slate-400 max-w-sm mx-auto">
            You haven't run any predictions yet in this browser session. Navigate to Live Diagnosis to test the model.
          </p>
          <Link
            to="/diagnosis"
            className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl bg-brand-600 hover:bg-brand-500 text-xs font-bold text-white shadow"
          >
            <Activity className="w-4 h-4" />
            Go to Live Diagnosis
          </Link>
        </div>
      ) : (
        <div className="glass-panel rounded-2xl p-6 border border-slate-800 shadow-xl space-y-4">
          {/* Search bar */}
          <div className="relative">
            <Search className="w-4 h-4 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search by dataset or diagnostic label..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full pl-9 pr-4 py-2 bg-slate-900/90 border border-slate-800 rounded-xl text-xs text-white placeholder-slate-500 focus:outline-none focus:border-brand-500 font-sans"
            />
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse font-mono">
              <thead>
                <tr className="border-b border-slate-800 bg-slate-900/60 text-slate-400 text-[11px]">
                  <th className="py-2.5 px-3">Timestamp</th>
                  <th className="py-2.5 px-3">Dataset</th>
                  <th className="py-2.5 px-3">Diagnostic Result</th>
                  <th className="py-2.5 px-3">Probability</th>
                  <th className="py-2.5 px-3">Confidence</th>
                  <th className="py-2.5 px-3">Route Executed</th>
                  <th className="py-2.5 px-3">Q-Layers</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {filteredHistory.map((item) => (
                  <tr key={item.id} className="hover:bg-slate-900/40">
                    <td className="py-2.5 px-3 text-slate-400 text-[11px] font-sans">{item.timestamp}</td>
                    <td className="py-2.5 px-3 text-slate-200">{item.dataset_name.split(' ')[0]}</td>
                    <td className="py-2.5 px-3 font-sans font-bold text-white">
                      <span className={`px-2 py-0.5 rounded-full text-[10px] ${
                        item.prediction_label.toLowerCase().includes('malignant') || item.prediction_label.toLowerCase().includes('present') || item.prediction_label.toLowerCase().includes('diabetic')
                          ? 'bg-rose-500/10 text-rose-400 border border-rose-500/30'
                          : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
                      }`}>
                        {item.prediction_label}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-slate-300">{(item.probability * 100).toFixed(1)}%</td>
                    <td className="py-2.5 px-3 text-brand-300 font-bold">{(item.confidence * 100).toFixed(1)}%</td>
                    <td className="py-2.5 px-3">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                        item.quantum_path === 0
                          ? 'bg-emerald-950 text-emerald-300'
                          : item.quantum_path === 1
                          ? 'bg-amber-950 text-amber-300'
                          : 'bg-purple-950 text-purple-300'
                      }`}>
                        Path {item.quantum_path}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-purple-400 font-bold">{item.quantum_layers} QL</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
