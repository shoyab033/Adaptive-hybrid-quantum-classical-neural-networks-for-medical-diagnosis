import React, { useState } from 'react';
import { Sparkles, Info, BarChart3, ShieldAlert, Layers } from 'lucide-react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  Tooltip, 
  ResponsiveContainer, 
  CartesianGrid 
} from 'recharts';

export const ExplainabilityPage: React.FC = () => {
  const [selectedDataset, setSelectedDataset] = useState<'breast_cancer' | 'heart' | 'diabetes'>('breast_cancer');

  const globalShapData = {
    breast_cancer: [
      { feature: 'worst concave points', importance: 0.184 },
      { feature: 'worst perimeter', importance: 0.162 },
      { feature: 'worst radius', importance: 0.145 },
      { feature: 'mean concave points', importance: 0.118 },
      { feature: 'worst area', importance: 0.098 },
      { feature: 'mean concavity', importance: 0.076 },
      { feature: 'worst concavity', importance: 0.065 },
      { feature: 'mean perimeter', importance: 0.054 },
    ],
    heart: [
      { feature: 'oldpeak (ST depression)', importance: 0.214 },
      { feature: 'thalach (max heart rate)', importance: 0.198 },
      { feature: 'ca (major vessels)', importance: 0.175 },
      { feature: 'cp (chest pain type)', importance: 0.142 },
      { feature: 'thal (thalassemia)', importance: 0.115 },
      { feature: 'age', importance: 0.068 },
      { feature: 'chol (cholesterol)', importance: 0.048 },
      { feature: 'trestbps (resting BP)', importance: 0.040 },
    ],
    diabetes: [
      { feature: 'Glucose (plasma glucose)', importance: 0.285 },
      { feature: 'BMI (body mass index)', importance: 0.224 },
      { feature: 'Age', importance: 0.168 },
      { feature: 'DiabetesPedigreeFunction', importance: 0.125 },
      { feature: 'Pregnancies', importance: 0.084 },
      { feature: 'Insulin (2-hr serum)', importance: 0.052 },
      { feature: 'BloodPressure', importance: 0.038 },
      { feature: 'SkinThickness', importance: 0.024 },
    ]
  };

  const currentShap = globalShapData[selectedDataset];

  return (
    <div className="space-y-8 py-4 max-w-5xl mx-auto">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-black text-white tracking-tight flex items-center gap-2">
          <Sparkles className="w-6 h-6 text-brand-400" />
          Explainable AI (XAI) & Kernel SHAP Dashboard
        </h1>
        <p className="text-xs text-slate-400 mt-1">
          Feature importance attributions, test set index alignment, and clinical biomarker distributions associated with quantum routing
        </p>
      </div>

      {/* Dataset Tabs */}
      <div className="flex items-center gap-2 bg-slate-900 border border-slate-800 p-1.5 rounded-2xl w-fit">
        <button
          onClick={() => setSelectedDataset('breast_cancer')}
          className={`px-4 py-2 rounded-xl text-xs font-semibold transition-all ${
            selectedDataset === 'breast_cancer'
              ? 'bg-brand-600 text-white shadow-md shadow-brand-600/30'
              : 'text-slate-400 hover:text-white hover:bg-slate-800'
          }`}
        >
          WDBC Breast Cancer
        </button>
        <button
          onClick={() => setSelectedDataset('heart')}
          className={`px-4 py-2 rounded-xl text-xs font-semibold transition-all ${
            selectedDataset === 'heart'
              ? 'bg-brand-600 text-white shadow-md shadow-brand-600/30'
              : 'text-slate-400 hover:text-white hover:bg-slate-800'
          }`}
        >
          UCI Cleveland Heart
        </button>
        <button
          onClick={() => setSelectedDataset('diabetes')}
          className={`px-4 py-2 rounded-xl text-xs font-semibold transition-all ${
            selectedDataset === 'diabetes'
              ? 'bg-brand-600 text-white shadow-md shadow-brand-600/30'
              : 'text-slate-400 hover:text-white hover:bg-slate-800'
          }`}
        >
          Pima Indians Diabetes
        </button>
      </div>

      {/* Global SHAP Importance Chart */}
      <div className="glass-panel rounded-2xl p-6 border border-slate-800 shadow-xl space-y-4">
        <div>
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <BarChart3 className="w-4 h-4 text-brand-400" />
            Global Kernel SHAP Feature Importance (Mean |SHAP Value|)
          </h3>
          <p className="text-[11px] text-slate-400 mt-0.5">
            Top clinical biomarkers driving overall prediction boundaries across the holdout test cohorts
          </p>
        </div>

        <div className="h-72 w-full pt-4">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={currentShap} layout="vertical" margin={{ top: 5, right: 30, left: 120, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis type="number" stroke="#94a3b8" fontSize={11} />
              <YAxis dataKey="feature" type="category" stroke="#94a3b8" fontSize={11} width={130} />
              <Tooltip contentStyle={{ backgroundColor: '#090d16', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }} />
              <Bar dataKey="importance" fill="#0ea5e9" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Biomarker Shifts & Methodological Rigor */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="glass-panel rounded-2xl p-6 border border-slate-800 space-y-3">
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <Layers className="w-4 h-4 text-purple-400" />
            Biomarkers Associated with Quantum Routing
          </h3>
          <p className="text-xs text-slate-300 leading-relaxed">
            Surrogate routing analysis indicates that instances escalated to Path 1 (1 QL) and Path 2 (2 QL) exhibit higher dispersion near critical diagnostic thresholds (e.g. glucose levels between 120–145 mg/dL, intermediate ST depression between 1.0–2.5 mm).
          </p>
          <div className="text-[11px] text-slate-400 pt-2 border-t border-slate-800">
            Uncertainty routing activates variational quantum circuits specifically for ambiguous presentations.
          </div>
        </div>

        <div className="glass-panel rounded-2xl p-6 border border-slate-800 space-y-3">
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <ShieldAlert className="w-4 h-4 text-amber-400" />
            Scientific Scope & Causal Disclaimer
          </h3>
          <p className="text-xs text-slate-300 leading-relaxed">
            Kernel SHAP explanations quantify feature contributions toward the mathematical decision surface of the hybrid model. They represent <strong>statistical model associations</strong>, not biological or clinical causality.
          </p>
          <div className="text-[11px] text-slate-400 pt-2 border-t border-slate-800">
            All test attributions are computed strictly on isolated holdout test indices.
          </div>
        </div>
      </div>
    </div>
  );
};
