import React, { useState, useRef, useEffect } from 'react';
import { NavLink, Link } from 'react-router-dom';
import { 
  Activity, 
  Cpu, 
  History, 
  Home, 
  ChevronDown, 
  Layers, 
  Sparkles, 
  Database, 
  Sliders, 
  HelpCircle,
  Menu,
  X,
  Stethoscope,
  Info
} from 'lucide-react';

export const Navbar: React.FC = () => {
  const [researchDropdownOpen, setResearchDropdownOpen] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown on outside click
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setResearchDropdownOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const researchItems = [
    { to: '/how-it-works', label: 'How It Works', icon: Info, desc: 'Plain English guide to adaptive quantum routing' },
    { to: '/architecture', label: 'Model Architecture', icon: Layers, desc: 'Classical encoder, confidence router, & VQC' },
    { to: '/circuit', label: 'Quantum Circuit', icon: Cpu, desc: 'Interactive 4-qubit parameterized circuit layout' },
    { to: '/explainability', label: 'XAI / SHAP Dashboard', icon: Sparkles, desc: 'Global Kernel SHAP feature importances' },
    { to: '/datasets', label: 'Dataset Explorer', icon: Database, desc: 'WDBC, Heart Disease, & Pima Diabetes' },
    { to: '/performance', label: 'Benchmark Performance', icon: Sliders, desc: 'Table II cross-validation & holdout metrics' },
    { to: '/workload', label: 'Quantum Workload Savings', icon: Cpu, desc: 'Table III estimated layer reduction metrics' },
    { to: '/about', label: 'About & Citation', icon: HelpCircle, desc: 'Project specs & academic BibTeX citation' },
  ];

  return (
    <header className="sticky top-0 z-50 bg-slate-950/95 backdrop-blur-md border-b border-slate-800/90 shadow-lg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          
          {/* Brand Logo */}
          <NavLink to="/" className="flex items-center space-x-3 group">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-600 via-blue-600 to-indigo-600 flex items-center justify-center shadow-lg shadow-blue-500/20 group-hover:scale-105 transition-transform">
              <Stethoscope className="w-5 h-5 text-white" />
            </div>
            <div>
              <div className="text-base font-extrabold tracking-tight text-white flex items-center gap-2">
                Adaptive <span className="text-cyan-400">HQNN</span>
                <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded-full bg-cyan-950 text-cyan-300 border border-cyan-800/60 font-bold">
                  Medical AI
                </span>
              </div>
              <p className="text-[11px] text-slate-400 font-medium">Early Disease Diagnosis Platform</p>
            </div>
          </NavLink>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center space-x-2">
            {/* Primary Link 1: Home */}
            <NavLink
              to="/"
              className={({ isActive }) =>
                `px-3.5 py-2 rounded-xl text-xs font-semibold transition-all flex items-center gap-1.5 ${
                  isActive
                    ? 'bg-slate-800 text-white shadow-sm'
                    : 'text-slate-300 hover:text-white hover:bg-slate-900'
                }`
              }
            >
              <Home className="w-4 h-4 text-slate-400" />
              Home
            </NavLink>

            {/* Primary Link 2: Predict Disease (Primary CTA) */}
            <NavLink
              to="/predict"
              className={({ isActive }) =>
                `px-4 py-2 rounded-xl text-xs font-bold transition-all flex items-center gap-1.5 shadow-md ${
                  isActive
                    ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-cyan-500/25 ring-2 ring-cyan-400'
                    : 'bg-gradient-to-r from-cyan-600 to-blue-700 hover:from-cyan-500 hover:to-blue-600 text-white shadow-blue-600/30'
                }`
              }
            >
              <Activity className="w-4 h-4 text-cyan-200 animate-pulse" />
              Predict Disease
            </NavLink>

            {/* Primary Link 3: Prediction History */}
            <NavLink
              to="/history"
              className={({ isActive }) =>
                `px-3.5 py-2 rounded-xl text-xs font-semibold transition-all flex items-center gap-1.5 ${
                  isActive
                    ? 'bg-slate-800 text-white shadow-sm'
                    : 'text-slate-300 hover:text-white hover:bg-slate-900'
                }`
              }
            >
              <History className="w-4 h-4 text-slate-400" />
              Prediction History
            </NavLink>

            {/* Secondary Dropdown: Research & Technical Details */}
            <div className="relative" ref={dropdownRef}>
              <button
                type="button"
                onClick={() => setResearchDropdownOpen(!researchDropdownOpen)}
                className={`px-3.5 py-2 rounded-xl text-xs font-semibold transition-all flex items-center gap-1.5 border border-slate-800 ${
                  researchDropdownOpen
                    ? 'bg-slate-800 text-white border-slate-700'
                    : 'text-slate-400 hover:text-white hover:bg-slate-900'
                }`}
              >
                <span>Research & Architecture</span>
                <ChevronDown className={`w-3.5 h-3.5 transition-transform duration-200 ${researchDropdownOpen ? 'rotate-180 text-cyan-400' : ''}`} />
              </button>

              {researchDropdownOpen && (
                <div className="absolute right-0 mt-2 w-72 bg-slate-900/95 border border-slate-800 rounded-2xl shadow-2xl p-2 z-50 backdrop-blur-xl animate-in fade-in slide-in-from-top-2 duration-150">
                  <div className="text-[10px] font-mono uppercase tracking-wider text-slate-400 px-3 py-1.5">
                    Technical Specifications
                  </div>
                  <div className="space-y-1">
                    {researchItems.map((item) => {
                      const Icon = item.icon;
                      return (
                        <NavLink
                          key={item.to}
                          to={item.to}
                          onClick={() => setResearchDropdownOpen(false)}
                          className={({ isActive }) =>
                            `flex items-start gap-2.5 p-2.5 rounded-xl transition-all ${
                              isActive
                                ? 'bg-cyan-500/15 text-cyan-300 border border-cyan-500/30'
                                : 'hover:bg-slate-800 text-slate-300 hover:text-white'
                            }`
                          }
                        >
                          <Icon className="w-4 h-4 text-cyan-400 mt-0.5 shrink-0" />
                          <div>
                            <div className="text-xs font-semibold leading-tight">{item.label}</div>
                            <div className="text-[10px] text-slate-400 leading-tight mt-0.5">{item.desc}</div>
                          </div>
                        </NavLink>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>

          </nav>

          {/* Mobile Menu Button */}
          <div className="flex md:hidden">
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="p-2 rounded-xl text-slate-400 hover:text-white hover:bg-slate-800"
            >
              {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </button>
          </div>

        </div>
      </div>

      {/* Mobile Drawer */}
      {mobileMenuOpen && (
        <div className="md:hidden bg-slate-950/98 border-b border-slate-800 px-4 pt-3 pb-6 space-y-2">
          <NavLink
            to="/"
            onClick={() => setMobileMenuOpen(false)}
            className="flex items-center gap-2 px-3.5 py-2.5 rounded-xl text-sm font-semibold text-slate-300 hover:bg-slate-900"
          >
            <Home className="w-4 h-4 text-cyan-400" />
            Home
          </NavLink>
          <NavLink
            to="/predict"
            onClick={() => setMobileMenuOpen(false)}
            className="flex items-center gap-2 px-3.5 py-2.5 rounded-xl text-sm font-bold bg-cyan-600 text-white shadow-md shadow-cyan-600/30"
          >
            <Activity className="w-4 h-4 text-white" />
            Predict Disease
          </NavLink>
          <NavLink
            to="/history"
            onClick={() => setMobileMenuOpen(false)}
            className="flex items-center gap-2 px-3.5 py-2.5 rounded-xl text-sm font-semibold text-slate-300 hover:bg-slate-900"
          >
            <History className="w-4 h-4 text-cyan-400" />
            Prediction History
          </NavLink>

          <div className="pt-3 border-t border-slate-800">
            <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider block px-3 mb-2">
              Research & Technical Specs
            </span>
            <div className="grid grid-cols-2 gap-1">
              {researchItems.map((item) => (
                <NavLink
                  key={item.to}
                  to={item.to}
                  onClick={() => setMobileMenuOpen(false)}
                  className="px-3 py-2 rounded-lg text-xs text-slate-400 hover:text-white hover:bg-slate-900"
                >
                  {item.label}
                </NavLink>
              ))}
            </div>
          </div>
        </div>
      )}
    </header>
  );
};
