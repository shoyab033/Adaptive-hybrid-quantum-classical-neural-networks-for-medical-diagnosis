import React, { useState, useEffect } from 'react';
import { getModelInfo } from '../services/api';
import { ModelInfoResponse } from '../types';
import { HelpCircle, BookOpen, Cpu, ShieldCheck, Code, Copy, Check } from 'lucide-react';

export const AboutPage: React.FC = () => {
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    const fetchInfo = async () => {
      try {
        const data = await getModelInfo();
        setModelInfo(data);
      } catch (e) {
        console.error(e);
      }
    };
    fetchInfo();
  }, []);

  const handleCopyBibtex = () => {
    if (modelInfo) {
      navigator.clipboard.writeText(modelInfo.citation_bibtex);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div className="space-y-10 py-4 max-w-4xl mx-auto">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-black text-white tracking-tight flex items-center gap-2">
          <HelpCircle className="w-6 h-6 text-brand-400" />
          About the Adaptive HQNN Research Project
        </h1>
        <p className="text-xs text-slate-400 mt-1">
          Academic project background, technical specification, citations, and research disclaimers
        </p>
      </div>

      {/* Abstract Card */}
      <div className="glass-panel rounded-2xl p-6 border border-slate-800 space-y-4 shadow-xl">
        <h3 className="text-sm font-bold text-white uppercase font-mono tracking-wider flex items-center gap-2">
          <BookOpen className="w-4 h-4 text-brand-400" />
          Project Abstract
        </h3>
        <p className="text-xs text-slate-300 leading-relaxed">
          Hybrid Quantum-Classical Neural Networks (HQNNs) have attracted substantial interest for clinical diagnostic applications on tabular medical data. However, conventional HQNN architectures impose a static circuit execution paradigm in which every input instance is subjected to the same fixed-depth parameterized quantum circuit, irrespective of sample difficulty or model uncertainty. In this research, we introduce an <strong>Adaptive Hybrid Quantum-Classical Neural Network (Adaptive HQNN)</strong> that dynamically allocates quantum processing based on instance-level prediction confidence derived from an initial classical feature encoder.
        </p>
      </div>

      {/* Technology Stack Grid */}
      <div className="glass-panel rounded-2xl p-6 border border-slate-800 space-y-4">
        <h3 className="text-sm font-bold text-white uppercase font-mono tracking-wider flex items-center gap-2">
          <Code className="w-4 h-4 text-purple-400" />
          Technical Stack & Dependencies
        </h3>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
            <span className="text-[10px] text-slate-400 block">Quantum Simulator</span>
            <span className="font-bold text-white mt-0.5 block">PennyLane (default.qubit)</span>
          </div>
          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
            <span className="text-[10px] text-slate-400 block">Deep Learning Engine</span>
            <span className="font-bold text-white mt-0.5 block">PyTorch 2.x</span>
          </div>
          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
            <span className="text-[10px] text-slate-400 block">Backend Framework</span>
            <span className="font-bold text-white mt-0.5 block">FastAPI & Uvicorn</span>
          </div>
          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
            <span className="text-[10px] text-slate-400 block">Frontend Framework</span>
            <span className="font-bold text-white mt-0.5 block">React 18 + Vite + TS</span>
          </div>
        </div>
      </div>

      {/* Citation Box */}
      {modelInfo && (
        <div className="glass-panel rounded-2xl p-6 border border-slate-800 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-white font-mono uppercase tracking-wider">
              BibTeX Citation
            </h3>
            <button
              onClick={handleCopyBibtex}
              className="px-3 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-mono flex items-center gap-1.5 transition-colors"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
              {copied ? 'Copied' : 'Copy'}
            </button>
          </div>
          <pre className="bg-slate-950 p-4 rounded-xl border border-slate-800 text-[11px] font-mono text-slate-300 overflow-x-auto">
            {modelInfo.citation_bibtex}
          </pre>
        </div>
      )}

      {/* Ethical & Regulatory Disclaimer */}
      <div className="bg-amber-500/10 border border-amber-500/30 rounded-2xl p-6 space-y-2 text-xs text-amber-200/90 leading-relaxed">
        <h4 className="font-bold text-amber-300 flex items-center gap-2 text-sm font-mono uppercase">
          <ShieldCheck className="w-4 h-4 text-amber-400" />
          Ethical & Clinical Research Disclaimer
        </h4>
        <p>
          This web platform is designed strictly for academic evaluation and experimental demonstration of quantum machine learning dynamic routing concepts. It is <strong>NOT a certified medical device</strong> and must not be used for clinical diagnosis, patient triage, or direct medical decision-making in healthcare facilities.
        </p>
      </div>

    </div>
  );
};
