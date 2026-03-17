import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { CheckCircle, XCircle, Star, ArrowLeft, User, Briefcase, Tag, Download, Cloud, Database, AlertCircle } from 'lucide-react';
import { colors } from '../config/colors';

interface CriterionScore {
  criterion_name: string;
  score: number;
  explanation: string;
}

interface Analysis {
  job_category: string;
  job_subcategory: string;
  criteria_scores: CriterionScore[];
  pros: string[];
  cons: string[];
  overall_assessment: string;
  key_requirements_met: string[];
  key_requirements_missing: string[];
  skills_identified: string[];
  skills_relevant: string[];
  skills_missing: string[];
}

interface Result {
  candidate_name: string;
  candidate_email: string;
  trade: string;
  ats_score: number;
  ai_score: number;
  analysis: Analysis;
  azure_url?: string;
  azure_error?: string;
  salesforce_record?: { success: boolean; result?: any; error?: string };
  pdf_generated?: boolean;
}

const ScoreRing: React.FC<{ score: number; max: number; label: string; color: string }> = ({ score, max, label, color }) => {
  const pct = Math.round((score / max) * 100);
  return (
    <div className="flex flex-col items-center gap-1">
      <div
        className="w-20 h-20 rounded-full flex items-center justify-center font-black text-xl text-white shadow-lg"
        style={{ backgroundColor: color }}
      >
        {pct}%
      </div>
      <span className="text-xs font-semibold text-gray-600 text-center">{label}</span>
    </div>
  );
};

const ScoreBar: React.FC<{ score: number; name: string }> = ({ score, name }) => {
  const pct = (score / 5) * 100;
  const color = score >= 4 ? '#22c55e' : score >= 3 ? colors.brand.blue : '#f97316';
  return (
    <div className="mb-3">
      <div className="flex justify-between text-xs font-medium text-gray-700 mb-1">
        <span>{name}</span>
        <span style={{ color }}>{score.toFixed(1)}/5</span>
      </div>
      <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
        <div className="h-full rounded-full transition-all" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
};

const Pill: React.FC<{ text: string; variant: 'green' | 'red' | 'blue' | 'gray' }> = ({ text, variant }) => {
  const styles = {
    green: 'bg-green-50 text-green-700 border-green-200',
    red: 'bg-red-50 text-red-700 border-red-200',
    blue: 'bg-blue-50 text-blue-700 border-blue-200',
    gray: 'bg-gray-100 text-gray-600 border-gray-200',
  };
  return (
    <span className={`inline-block border rounded-full px-2.5 py-0.5 text-xs font-medium mr-1.5 mb-1.5 ${styles[variant]}`}>
      {text}
    </span>
  );
};

const AnalysisResultPage: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const result = location.state?.result as Result | undefined;

  if (!result) {
    return (
      <div className="p-6 text-center">
        <p className="text-gray-500 mb-4">No analysis result found. Please upload a CV first.</p>
        <button
          onClick={() => navigate('/upload')}
          className="px-6 py-2 rounded-lg text-white font-semibold"
          style={{ backgroundColor: colors.brand.blue }}
        >
          Upload CV
        </button>
      </div>
    );
  }

  const { candidate_name, candidate_email, trade, ats_score, ai_score, analysis } = result;

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-6">
      {/* Back + Header */}
      <div className="flex items-center gap-3 mb-2">
        <button
          onClick={() => navigate('/upload')}
          className="flex items-center gap-1.5 text-sm font-medium text-gray-500 hover:text-gray-800 transition-all"
        >
          <ArrowLeft size={16} /> Back
        </button>
      </div>

      {/* Candidate Info */}
      <div className="bg-white rounded-xl p-5 border border-gray-100 shadow-sm flex flex-wrap gap-6 items-center">
        <div className="flex items-center gap-2 text-sm text-gray-700">
          <User size={16} style={{ color: colors.brand.blue }} />
          <span className="font-semibold">{candidate_name}</span>
          <span className="text-gray-400">·</span>
          <span className="text-gray-500">{candidate_email}</span>
        </div>
        <div className="flex items-center gap-2 text-sm text-gray-700">
          <Briefcase size={16} style={{ color: colors.brand.blue }} />
          <span>{trade}</span>
        </div>
        <div className="flex items-center gap-2 text-sm text-gray-700">
          <Tag size={16} style={{ color: colors.brand.blue }} />
          <span>{analysis.job_category} — {analysis.job_subcategory}</span>
        </div>
      </div>

      {/* Score Cards */}
      <div className="bg-white rounded-xl p-6 border border-gray-100 shadow-sm">
        <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-5">Scores</h3>
        <div className="flex flex-wrap gap-8 justify-center">
          <ScoreRing score={ats_score} max={100} label="ATS Keyword Score" color={ats_score >= 60 ? '#22c55e' : ats_score >= 40 ? colors.brand.blue : '#f97316'} />
          <ScoreRing score={ai_score} max={100} label="AI Evaluation Score" color={ai_score >= 80 ? '#22c55e' : ai_score >= 60 ? colors.brand.blue : '#f97316'} />
        </div>
      </div>

      {/* Overall Assessment */}
      <div className="bg-white rounded-xl p-6 border border-gray-100 shadow-sm">
        <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-3">Overall Assessment</h3>
        <p className="text-gray-700 text-sm leading-relaxed">{analysis.overall_assessment}</p>
      </div>

      {/* Storage & Integration Notifications */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* PDF Report Section */}
        {result.pdf_generated ? (
          <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-xl p-5 border border-blue-200 shadow-sm">
            <div className="flex items-start gap-3">
              <Download size={20} className="text-blue-600 mt-0.5 shrink-0" />
              <div className="flex-1">
                <h3 className="text-sm font-bold text-blue-900 mb-1">📄 PDF Report Generated</h3>
                <p className="text-xs text-blue-700 mb-3">Professional analysis report ready</p>
                {result.azure_url ? (
                  <a
                    href={result.azure_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    download
                    className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 text-white text-xs font-semibold rounded-lg hover:bg-blue-700 transition-all"
                  >
                    <Download size={14} /> Download PDF
                  </a>
                ) : (
                  <span className="text-xs text-blue-600">PDF generated (waiting for Azure upload...)</span>
                )}
              </div>
            </div>
          </div>
        ) : (
          <div className="bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl p-5 border border-gray-200 shadow-sm">
            <div className="flex items-start gap-3">
              <AlertCircle size={20} className="text-gray-400 mt-0.5 shrink-0" />
              <div className="flex-1">
                <h3 className="text-sm font-bold text-gray-600 mb-1">📄 PDF Report</h3>
                <p className="text-xs text-gray-500">PDF generation not available</p>
              </div>
            </div>
          </div>
        )}

        {/* Azure Storage Section */}
        {result.azure_url ? (
          <div className="bg-gradient-to-br from-green-50 to-green-100 rounded-xl p-5 border border-green-200 shadow-sm">
            <div className="flex items-start gap-3">
              <Cloud size={20} className="text-green-600 mt-0.5 shrink-0" />
              <div className="flex-1">
                <h3 className="text-sm font-bold text-green-900 mb-1">✓ Stored in Azure</h3>
                <p className="text-xs text-green-700 mb-2">PDF securely uploaded to cloud storage</p>
                <a
                  href={result.azure_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-3 py-1.5 bg-green-600 text-white text-xs font-semibold rounded-lg hover:bg-green-700 transition-all"
                >
                  View in Azure →
                </a>
              </div>
            </div>
          </div>
        ) : result.azure_error ? (
          <div className="bg-gradient-to-br from-yellow-50 to-yellow-100 rounded-xl p-5 border border-yellow-200 shadow-sm">
            <div className="flex items-start gap-3">
              <AlertCircle size={20} className="text-yellow-600 mt-0.5 shrink-0" />
              <div className="flex-1">
                <h3 className="text-sm font-bold text-yellow-900 mb-1">⚠️ Azure Upload Failed</h3>
                <p className="text-xs text-yellow-700 mb-2 font-mono">{result.azure_error}</p>
                <p className="text-xs text-yellow-600">Check your Azure credentials in .env file or see AZURE_SETUP.md for help</p>
              </div>
            </div>
          </div>
        ) : (
          <div className="bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl p-5 border border-gray-200 shadow-sm">
            <div className="flex items-start gap-3">
              <AlertCircle size={20} className="text-gray-400 mt-0.5 shrink-0" />
              <div className="flex-1">
                <h3 className="text-sm font-bold text-gray-600 mb-1">☁️ Azure Storage</h3>
                <p className="text-xs text-gray-500">Configuration not available</p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Salesforce Integration */}
      {result.salesforce_record && (
        <div className={`rounded-xl p-5 border shadow-sm ${
          result.salesforce_record.success
            ? 'bg-gradient-to-br from-emerald-50 to-emerald-100 border-emerald-200'
            : 'bg-gradient-to-br from-red-50 to-red-100 border-red-200'
        }`}>
          <div className="flex items-start gap-3">
            <Database size={20} className={`${result.salesforce_record.success ? 'text-emerald-600' : 'text-red-600'} mt-0.5 shrink-0`} />
            <div className="flex-1">
              <h3 className={`text-sm font-bold ${result.salesforce_record.success ? 'text-emerald-900' : 'text-red-900'} mb-1`}>
                {result.salesforce_record.success ? '✓ Saved to Salesforce' : '✗ Salesforce Error'}
              </h3>
              {result.salesforce_record.success ? (
                <>
                  <p className={`text-xs ${result.salesforce_record.success ? 'text-emerald-700' : 'text-red-700'} mb-2`}>
                    Engineer Application record created successfully
                  </p>
                  {result.salesforce_record.result?.id && (
                    <p className={`text-xs font-mono ${result.salesforce_record.success ? 'text-emerald-600' : 'text-red-600'}`}>
                      📌 Record ID: {result.salesforce_record.result.id}
                    </p>
                  )}
                </>
              ) : (
                <p className={`text-xs ${result.salesforce_record.success ? 'text-emerald-700' : 'text-red-700'}`}>
                  {result.salesforce_record.error || 'Could not create Salesforce record'}
                </p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Criteria Scores */}
      {analysis.criteria_scores?.length > 0 && (
        <div className="bg-white rounded-xl p-6 border border-gray-100 shadow-sm">
          <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-4">Evaluation Criteria</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8">
            {analysis.criteria_scores.map((c, i) => (
              <div key={i}>
                <ScoreBar score={c.score} name={c.criterion_name} />
                <p className="text-xs text-gray-500 mb-4 -mt-1">{c.explanation}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Pros & Cons */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-white rounded-xl p-5 border border-gray-100 shadow-sm">
          <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-3 flex items-center gap-1.5">
            <CheckCircle size={14} className="text-green-500" /> Strengths
          </h3>
          <ul className="space-y-1.5">
            {analysis.pros.map((p, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-gray-700">
                <span className="text-green-500 mt-0.5 shrink-0">✓</span>{p}
              </li>
            ))}
          </ul>
        </div>
        <div className="bg-white rounded-xl p-5 border border-gray-100 shadow-sm">
          <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-3 flex items-center gap-1.5">
            <XCircle size={14} className="text-red-400" /> Weaknesses
          </h3>
          <ul className="space-y-1.5">
            {analysis.cons.map((c, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-gray-700">
                <span className="text-red-400 mt-0.5 shrink-0">✗</span>{c}
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* Requirements */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-white rounded-xl p-5 border border-gray-100 shadow-sm">
          <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-3">Requirements Met</h3>
          <ul className="space-y-1.5">
            {analysis.key_requirements_met.map((r, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-gray-700">
                <span className="text-green-500 mt-0.5 shrink-0">✓</span>{r}
              </li>
            ))}
          </ul>
        </div>
        <div className="bg-white rounded-xl p-5 border border-gray-100 shadow-sm">
          <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-3">Requirements Missing</h3>
          <ul className="space-y-1.5">
            {analysis.key_requirements_missing.map((r, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-gray-700">
                <span className="text-red-400 mt-0.5 shrink-0">✗</span>{r}
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* Skills */}
      <div className="bg-white rounded-xl p-6 border border-gray-100 shadow-sm">
        <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-4 flex items-center gap-1.5">
          <Star size={14} style={{ color: colors.brand.blue }} /> Skills Analysis
        </h3>
        <div className="space-y-4">
          <div>
            <p className="text-xs font-semibold text-gray-500 mb-2">All Identified Skills</p>
            <div>{analysis.skills_identified.map((s, i) => <Pill key={i} text={s} variant="gray" />)}</div>
          </div>
          <div>
            <p className="text-xs font-semibold text-gray-500 mb-2">Relevant to Job</p>
            <div>{analysis.skills_relevant.map((s, i) => <Pill key={i} text={s} variant="blue" />)}</div>
          </div>
          {analysis.skills_missing.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-gray-500 mb-2">Missing Skills</p>
              <div>{analysis.skills_missing.map((s, i) => <Pill key={i} text={s} variant="red" />)}</div>
            </div>
          )}
        </div>
      </div>

      {/* Analyse Another */}
      <div className="text-center pb-4">
        <button
          onClick={() => navigate('/upload')}
          className="px-8 py-3 rounded-xl text-white font-bold text-sm transition-all hover:opacity-90"
          style={{ backgroundColor: colors.brand.blue }}
        >
          Analyse Another CV
        </button>
      </div>
    </div>
  );
};

export default AnalysisResultPage;
