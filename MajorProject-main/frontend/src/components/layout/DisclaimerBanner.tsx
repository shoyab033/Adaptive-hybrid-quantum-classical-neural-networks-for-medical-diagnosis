import React from 'react';
import { AlertTriangle } from 'lucide-react';

export const DisclaimerBanner: React.FC = () => {
  return (
    <div className="bg-amber-500/10 border-b border-amber-500/20 px-4 py-2 text-xs text-amber-200/90">
      <div className="max-w-7xl mx-auto flex items-center justify-center gap-2 text-center">
        <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
        <span>
          <strong>Research Prototype:</strong> This application is intended for educational and research purposes only. It does not provide medical diagnosis or replace consultation with a qualified healthcare professional.
        </span>
      </div>
    </div>
  );
};
