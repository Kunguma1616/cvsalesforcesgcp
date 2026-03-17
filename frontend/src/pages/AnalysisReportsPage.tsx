import React from 'react';
import { useNavigate } from 'react-router-dom';
import { FileText, Upload } from 'lucide-react';
import { colors } from '../config/colors';

const AnalysisReportsPage: React.FC = () => {
  const navigate = useNavigate();

  return (
    <div className="p-6 max-w-3xl mx-auto">
      <div className="mb-6">
        <h2 className="text-2xl font-bold mb-1" style={{ color: colors.brand.blue }}>Analysis Reports</h2>
        <p className="text-gray-500 text-sm">View past CV analysis results and reports.</p>
      </div>

      <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-12 text-center">
        <FileText size={48} className="mx-auto mb-4" style={{ color: colors.brand.blue, opacity: 0.3 }} />
        <p className="text-gray-500 font-medium mb-1">No reports yet</p>
        <p className="text-gray-400 text-sm mb-6">Upload and analyse a CV to see results here.</p>
        <button
          onClick={() => navigate('/upload')}
          className="inline-flex items-center gap-2 px-6 py-2.5 rounded-xl text-white font-semibold text-sm hover:opacity-90 transition-all"
          style={{ backgroundColor: colors.brand.blue }}
        >
          <Upload size={16} />
          Upload CV
        </button>
      </div>
    </div>
  );
};

export default AnalysisReportsPage;
