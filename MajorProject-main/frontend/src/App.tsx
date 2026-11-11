import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Navbar } from './components/layout/Navbar';
import { Footer } from './components/layout/Footer';
import { DisclaimerBanner } from './components/layout/DisclaimerBanner';

import { LandingPage } from './pages/LandingPage';
import { PredictionPage } from './pages/PredictionPage';
import { HowItWorksPage } from './pages/HowItWorksPage';
import { PredictionHistoryPage } from './pages/PredictionHistoryPage';
import { ModelArchitecturePage } from './pages/ModelArchitecturePage';
import { QuantumCircuitPage } from './pages/QuantumCircuitPage';
import { ExplainabilityPage } from './pages/ExplainabilityPage';
import { DatasetsPage } from './pages/DatasetsPage';
import { PerformancePage } from './pages/PerformancePage';
import { QuantumWorkloadPage } from './pages/QuantumWorkloadPage';
import { AboutPage } from './pages/AboutPage';

export const App: React.FC = () => {
  return (
    <BrowserRouter>
      <div className="flex flex-col min-h-screen bg-slate-950 text-slate-100 font-sans selection:bg-cyan-500 selection:text-white">
        <DisclaimerBanner />
        <Navbar />
        <main className="flex-grow max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/predict" element={<PredictionPage />} />
            <Route path="/diagnosis" element={<PredictionPage />} />
            <Route path="/how-it-works" element={<HowItWorksPage />} />
            <Route path="/history" element={<PredictionHistoryPage />} />
            <Route path="/architecture" element={<ModelArchitecturePage />} />
            <Route path="/circuit" element={<QuantumCircuitPage />} />
            <Route path="/explainability" element={<ExplainabilityPage />} />
            <Route path="/datasets" element={<DatasetsPage />} />
            <Route path="/performance" element={<PerformancePage />} />
            <Route path="/workload" element={<QuantumWorkloadPage />} />
            <Route path="/about" element={<AboutPage />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </BrowserRouter>
  );
};

export default App;
