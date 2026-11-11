import React from 'react';
import { Cpu, ShieldCheck, Github, BookOpen } from 'lucide-react';
import { Link } from 'react-router-dom';

export const Footer: React.FC = () => {
  return (
    <footer className="bg-slate-950 border-t border-slate-800/80 pt-12 pb-8 mt-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-10">
          {/* Brand Col */}
          <div className="space-y-4 md:col-span-2">
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 rounded-lg bg-brand-600 flex items-center justify-center">
                <Cpu className="w-4 h-4 text-white" />
              </div>
              <span className="text-base font-bold text-white tracking-tight">Adaptive HQNN Platform</span>
            </div>
            <p className="text-xs text-slate-400 max-w-md leading-relaxed">
              An Explainable Quantum Machine Learning research platform introducing dynamic quantum layer selection for medical tabular disease prediction across oncology, cardiology, and endocrinology.
            </p>
            <div className="flex items-center gap-3 text-xs text-slate-400">
              <span className="flex items-center gap-1">
                <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
                Zero Data Leakage (Nested 3-Fold Inner CV)
              </span>
            </div>
          </div>

          {/* Quick Links */}
          <div>
            <h4 className="text-xs font-semibold text-white uppercase tracking-wider mb-3 font-mono">Platform Pages</h4>
            <ul className="space-y-2 text-xs text-slate-400">
              <li><Link to="/diagnosis" className="hover:text-brand-400 transition-colors">Live Diagnosis</Link></li>
              <li><Link to="/dashboard" className="hover:text-brand-400 transition-colors">Research Dashboard</Link></li>
              <li><Link to="/architecture" className="hover:text-brand-400 transition-colors">Model Architecture</Link></li>
              <li><Link to="/circuit" className="hover:text-brand-400 transition-colors">Quantum Circuit Viewer</Link></li>
              <li><Link to="/explainability" className="hover:text-brand-400 transition-colors">Kernel SHAP Analysis</Link></li>
              <li><Link to="/workload" className="hover:text-brand-400 transition-colors">Quantum Workload Savings</Link></li>
            </ul>
          </div>

          {/* Research Datasets */}
          <div>
            <h4 className="text-xs font-semibold text-white uppercase tracking-wider mb-3 font-mono">Clinical Modalities</h4>
            <ul className="space-y-2 text-xs text-slate-400">
              <li><span className="text-slate-300 font-medium">WDBC Breast Cancer</span> (30 features)</li>
              <li><span className="text-slate-300 font-medium">UCI Cleveland Heart</span> (13 features)</li>
              <li><span className="text-slate-300 font-medium">Pima Indians Diabetes</span> (8 features)</li>
              <li className="pt-2 text-[11px] text-slate-500 font-mono">Built with PyTorch & PennyLane</li>
            </ul>
          </div>
        </div>

        <div className="border-t border-slate-800/60 pt-6 flex flex-col sm:flex-row items-center justify-between text-xs text-slate-500">
          <p>© 2026 Adaptive HQNN Research Team. All rights reserved.</p>
          <p className="mt-2 sm:mt-0 font-mono text-[11px]">
            Trained on 4-Qubit Variational Quantum Circuits (State-Vector Simulator)
          </p>
        </div>
      </div>
    </footer>
  );
};
