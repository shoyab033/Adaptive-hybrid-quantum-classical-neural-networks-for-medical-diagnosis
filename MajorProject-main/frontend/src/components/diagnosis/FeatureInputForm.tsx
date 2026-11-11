import React, { useState } from 'react';
import { DatasetMetadata, FeatureInfo, PresetSample } from '../../types';
import { 
  Sliders, 
  Sparkles, 
  RotateCcw, 
  Search, 
  Activity, 
  Check, 
  Info,
  Layers
} from 'lucide-react';

interface Props {
  dataset: DatasetMetadata;
  values: Record<string, number>;
  onChange: (featureName: string, val: number) => void;
  onSelectPreset: (preset: PresetSample) => void;
  onResetDefaults: () => void;
  onSubmit: () => void;
  loading: boolean;
}

export const FeatureInputForm: React.FC<Props> = ({
  dataset,
  values,
  onChange,
  onSelectPreset,
  onResetDefaults,
  onSubmit,
  loading,
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [activePresetId, setActivePresetId] = useState<string | null>(null);

  const filteredFeatures = dataset.features.filter((f) =>
    f.display_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    f.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handlePresetClick = (preset: PresetSample) => {
    setActivePresetId(preset.id);
    onSelectPreset(preset);
  };

  return (
    <div className="glass-panel rounded-2xl p-6 border border-slate-800 shadow-xl">
      {/* Header & Presets */}
      <div className="space-y-4 pb-6 border-b border-slate-800">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <Sliders className="w-5 h-5 text-brand-400" />
              Clinical Feature Parameters ({dataset.features.length} Features)
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Enter patient measurements or load one of the calibrated clinical sample presets below
            </p>
          </div>

          <button
            onClick={() => {
              setActivePresetId(null);
              onResetDefaults();
            }}
            className="self-start sm:self-auto px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-700 text-xs text-slate-300 font-medium transition-colors flex items-center gap-1.5"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            Reset Defaults
          </button>
        </div>

        {/* Preset Selector Buttons */}
        <div className="space-y-2">
          <span className="text-[11px] font-mono text-brand-300 uppercase tracking-wider font-semibold block">
            ⭐ Instant Clinical Presets (Live Testing)
          </span>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
            {dataset.presets.map((preset) => (
              <button
                key={preset.id}
                type="button"
                onClick={() => handlePresetClick(preset)}
                className={`p-3 rounded-xl text-left border transition-all text-xs ${
                  activePresetId === preset.id
                    ? 'bg-brand-500/20 border-brand-400 text-white shadow-md shadow-brand-500/10'
                    : 'bg-slate-900/80 border-slate-800 text-slate-300 hover:bg-slate-800/80 hover:border-slate-700'
                }`}
              >
                <div className="font-bold text-white flex items-center justify-between">
                  <span>{preset.name}</span>
                  {activePresetId === preset.id && <Check className="w-3.5 h-3.5 text-brand-400" />}
                </div>
                <p className="text-[10px] text-slate-400 mt-1 line-clamp-2 leading-relaxed">
                  {preset.description}
                </p>
                <span className="inline-block mt-2 font-mono text-[9px] px-1.5 py-0.5 rounded bg-slate-950 text-brand-300 border border-brand-900/60">
                  Target: {preset.expected_path}
                </span>
              </button>
            ))}
          </div>
        </div>

        {/* Feature Search Box (if more than 10 features) */}
        {dataset.features.length > 10 && (
          <div className="relative">
            <Search className="w-4 h-4 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder={`Search ${dataset.features.length} features (e.g. radius, perimeter, concave)...`}
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-9 pr-4 py-2 bg-slate-900/90 border border-slate-800 rounded-xl text-xs text-white placeholder-slate-500 focus:outline-none focus:border-brand-500 transition-colors font-sans"
            />
          </div>
        )}
      </div>

      {/* Inputs Grid */}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          onSubmit();
        }}
        className="mt-6 space-y-6"
      >
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 max-h-[500px] overflow-y-auto pr-1">
          {filteredFeatures.map((feat) => {
            const val = values[feat.name] !== undefined ? values[feat.name] : feat.default_value;
            return (
              <div key={feat.name} className="bg-slate-900/70 border border-slate-800 rounded-xl p-3.5 space-y-1.5 focus-within:border-brand-500/50 transition-colors">
                <div className="flex items-center justify-between">
                  <label htmlFor={feat.name} className="text-xs font-semibold text-slate-200 truncate pr-2" title={feat.description}>
                    {feat.display_name}
                  </label>
                  {feat.unit && (
                    <span className="text-[10px] font-mono text-slate-400 shrink-0">
                      {feat.unit}
                    </span>
                  )}
                </div>

                <div className="flex items-center gap-2">
                  <input
                    id={feat.name}
                    type="number"
                    step={feat.step || 0.01}
                    min={feat.min_value}
                    max={feat.max_value}
                    value={val}
                    onChange={(e) => {
                      setActivePresetId(null);
                      onChange(feat.name, parseFloat(e.target.value) || 0);
                    }}
                    className="w-full bg-slate-950 border border-slate-700/80 rounded-lg px-2.5 py-1.5 text-xs text-white font-mono focus:outline-none focus:border-brand-400"
                  />
                </div>

                <div className="flex items-center justify-between text-[10px] text-slate-400 font-mono">
                  <span>Min: {feat.min_value}</span>
                  <span>Max: {feat.max_value}</span>
                </div>
              </div>
            );
          })}
        </div>

        {/* Submit Button */}
        <div className="pt-4 border-t border-slate-800 flex items-center justify-end">
          <button
            type="submit"
            disabled={loading}
            className="w-full sm:w-auto px-8 py-3 rounded-xl bg-gradient-to-r from-brand-600 via-brand-500 to-quantum-600 hover:from-brand-500 hover:to-quantum-500 text-white font-bold text-sm shadow-lg shadow-brand-600/30 transition-all flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Activity className="w-4 h-4 text-white animate-pulse" />
            {loading ? 'Executing Adaptive HQNN (PennyLane Simulator)...' : 'Run Adaptive HQNN Prediction'}
          </button>
        </div>
      </form>
    </div>
  );
};
