import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { Upload, FileText, User, Mail, Briefcase, AlignLeft, AlertCircle } from 'lucide-react';
import { colors } from '../config/colors';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const TRADES = [
  'Access',
  'Brickwork & Paving',
  'Carpentry',
  'Drainage',
  'Drainage Survey',
  'Drains & Blockages (Soil Water)',
  'Drains & Blockages (Waste Water)',
  'Electrical',
  'Electrical Testing',
  'Fencing, Decking & Cladding',
  'Gardening',
  'Gas',
  'Gas Commercial',
  'Glazing',
  'Heating, Ventilation, & Air Conditioning',
  'Leak Detection - Drainage',
  'Leak Detection - Heating/Hot Water',
  'Leak Detection - Multi',
  'Leak Detection - Plumbing',
  'Leak Detection - Roofing',
  'Locksmith',
  'Multi Skilled',
  'Painting & Decorating',
  'Pest Control',
  'Plastering',
  'Plumbing',
  'Project Manager',
  'Roofing',
  'Tiling',
  'Ventilation',
  'Waste Clearance',
  'Windows & Doors',
];

const UploadCVPage: React.FC = () => {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [form, setForm] = useState({
    name: '',
    email: '',
    trade: '',
    job_description: '',
  });
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleFileDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const dropped = e.dataTransfer.files[0];
    if (dropped && dropped.type === 'application/pdf') {
      setFile(dropped);
      setError('');
    } else {
      setError('Only PDF files are accepted.');
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0];
    if (selected) {
      setFile(selected);
      setError('');
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) { setError('Please upload a PDF resume.'); return; }
    if (!form.name || !form.email || !form.trade || !form.job_description) {
      setError('Please fill in all fields.');
      return;
    }

    setLoading(true);
    setError('');

    const data = new FormData();
    data.append('name', form.name);
    data.append('email', form.email);
    data.append('trade', form.trade);
    data.append('job_description', form.job_description);
    data.append('resume', file);

    try {
      const res = await axios.post(`${API_URL}/cv/upload-and-analyze`, data, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      navigate('/analysis-result', { state: { result: res.data } });
    } catch (err: any) {
      const msg = err.response?.data?.detail ?? err.message ?? 'Analysis failed. Please try again.';
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header Banner */}
    <div
      style={{
        position: "relative",
        backgroundImage: "url(/aspectbackground.png)",
        backgroundSize: "cover",
        backgroundPosition: "center",
        height: "180px",
      }}
    >
      {/* Overlay */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          backgroundColor: "rgba(0,0,0,0.55)",
        }}
      />

      <div className="relative z-10 max-w-3xl mx-auto px-6 py-10">
        <h1 className="text-3xl font-bold text-white">
          Chumely, Upload CV for Analysis
        </h1>
        <p className="text-white/80 text-sm mt-2">
          Fill in candidate details and upload a PDF resume for AI-powered analysis.
        </p>
      </div>
    </div>

    {/* Form Section */}
    <div className="p-6 max-w-3xl mx-auto">
      <div className="mb-6">
        <h2 className="text-2xl font-bold mb-1" style={{ color: colors.brand.blue }}>Upload CV for Analysis</h2>
        <p className="text-gray-500 text-sm">Fill in candidate details and upload a PDF resume for AI-powered analysis.</p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-5">
        {/* Name + Email */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-1">
              <User size={14} className="inline mr-1" />Candidate Name
            </label>
            <input
              type="text"
              name="name"
              value={form.name}
              onChange={handleChange}
              placeholder="John Smith"
              className="w-full border border-gray-300 rounded-lg px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:border-transparent"
              style={{ '--tw-ring-color': colors.brand.blue } as React.CSSProperties}
              required
            />
          </div>
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-1">
              <Mail size={14} className="inline mr-1" />Email Address
            </label>
            <input
              type="email"
              name="email"
              value={form.email}
              onChange={handleChange}
              placeholder="john@example.com"
              className="w-full border border-gray-300 rounded-lg px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:border-transparent"
              required
            />
          </div>
        </div>

        {/* Trade */}
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-1">
            <Briefcase size={14} className="inline mr-1" />Trade / Profession
          </label>
          <select
            name="trade"
            value={form.trade}
            onChange={handleChange}
            className="w-full border border-gray-300 rounded-lg px-3 py-2.5 text-sm focus:outline-none focus:ring-2 bg-white"
            required
          >
            <option value="">Select a trade...</option>
            {TRADES.map((t) => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </div>

        {/* Job Description */}
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-1">
            <AlignLeft size={14} className="inline mr-1" />Job Description
          </label>
          <textarea
            name="job_description"
            value={form.job_description}
            onChange={handleChange}
            rows={6}
            placeholder="Paste the full job description here..."
            className="w-full border border-gray-300 rounded-lg px-3 py-2.5 text-sm focus:outline-none focus:ring-2 resize-none"
            required
          />
        </div>

        {/* PDF Upload */}
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-1">
            <FileText size={14} className="inline mr-1" />Resume (PDF only)
          </label>
          <div
            onDrop={handleFileDrop}
            onDragOver={(e) => e.preventDefault()}
            onClick={() => fileInputRef.current?.click()}
            className="border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all hover:bg-blue-50"
            style={{ borderColor: file ? colors.brand.blue : '#d1d5db' }}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf"
              onChange={handleFileSelect}
              className="hidden"
            />
            {file ? (
              <div className="flex items-center justify-center gap-2" style={{ color: colors.brand.blue }}>
                <FileText size={22} />
                <span className="font-semibold text-sm">{file.name}</span>
                <span className="text-xs text-gray-400">({(file.size / 1024).toFixed(0)} KB)</span>
              </div>
            ) : (
              <div className="text-gray-400">
                <Upload size={32} className="mx-auto mb-2" />
                <p className="text-sm font-medium">Drag & drop PDF here, or click to browse</p>
                <p className="text-xs mt-1">PDF files only</p>
              </div>
            )}
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="flex items-start gap-2 bg-red-50 border border-red-200 rounded-lg px-4 py-3 text-red-700 text-sm">
            <AlertCircle size={16} className="mt-0.5 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Submit */}
        <button
          type="submit"
          disabled={loading}
          className="w-full py-3 rounded-xl font-bold text-white text-sm transition-all hover:opacity-90 disabled:opacity-60 disabled:cursor-not-allowed"
          style={{ backgroundColor: colors.brand.blue }}
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
              </svg>
              Analysing CV... This may take 15–30 seconds
            </span>
          ) : (
            'Analyse CV'
          )}
        </button>
      </form>
    </div>
    </div>
  );
};

export default UploadCVPage;
