import React, { useState, useEffect, useMemo } from 'react';
import axios from 'axios';
import { Search, Trophy, Users, AlertCircle, ChevronUp, ChevronDown, Droplets, Zap } from 'lucide-react';
import { colors } from '../config/colors';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface Engineer {
  rank: number;
  id: string;
  name: string;
  email: string;
  trade_group: string;
  trade_type: 'WET' | 'DRY' | 'N/A';
}

type SortField = 'rank' | 'name' | 'trade_group' | 'trade_type';
type SortDir = 'asc' | 'desc';

function RankBadge({ rank }: { rank: number }) {
  if (rank === 1)
    return (
      <span className="inline-flex items-center gap-1 font-bold" style={{ color: '#b45309' }}>
        <Trophy size={14} /> 1
      </span>
    );
  if (rank === 2)
    return (
      <span className="inline-flex items-center gap-1 font-bold" style={{ color: '#6b7280' }}>
        <Trophy size={14} /> 2
      </span>
    );
  if (rank === 3)
    return (
      <span className="inline-flex items-center gap-1 font-bold" style={{ color: '#92400e' }}>
        <Trophy size={14} /> 3
      </span>
    );
  return <span className="text-gray-600 font-medium">{rank}</span>;
}

function TradeTypeBadge({ type }: { type: string }) {
  if (type === 'WET')
    return (
      <span
        className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-semibold"
        style={{ backgroundColor: '#dbeafe', color: '#1d4ed8' }}
      >
        <Droplets size={11} /> WET
      </span>
    );
  if (type === 'DRY')
    return (
      <span
        className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-semibold"
        style={{ backgroundColor: '#fef9c3', color: '#854d0e' }}
      >
        <Zap size={11} /> DRY
      </span>
    );
  return (
    <span
      className="px-2 py-0.5 rounded-full text-xs font-semibold"
      style={{ backgroundColor: '#f3f4f6', color: '#6b7280' }}
    >
      —
    </span>
  );
}

export const EngineerRankingPage: React.FC = () => {
  const [engineers, setEngineers] = useState<Engineer[]>([]);
  const [tradeGroups, setTradeGroups] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [nameFilter, setNameFilter] = useState('');
  const [tradeFilter, setTradeFilter] = useState('all');
  const [sortField, setSortField] = useState<SortField>('rank');
  const [sortDir, setSortDir] = useState<SortDir>('asc');

  useEffect(() => {
    axios
      .get(`${API_BASE}/api/ranking/trade-groups`)
      .then((res) => setTradeGroups(res.data.trade_groups || []))
      .catch(() => setTradeGroups([]));
  }, []);

  useEffect(() => {
    setLoading(true);
    setError(null);
    const params: Record<string, string> = {};
    if (tradeFilter !== 'all') params.trade_group = tradeFilter;
    axios
      .get(`${API_BASE}/api/ranking/engineers`, { params })
      .then((res) => setEngineers(res.data.engineers || []))
      .catch((err) => setError(err.response?.data?.detail || 'Failed to load engineer rankings.'))
      .finally(() => setLoading(false));
  }, [tradeFilter]);

  const handleSort = (field: SortField) => {
    if (sortField === field) setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    else { setSortField(field); setSortDir('asc'); }
  };

  const filtered = useMemo(() => {
    let list = engineers;
    if (nameFilter.trim()) {
      const q = nameFilter.toLowerCase();
      list = list.filter(
        (e) => e.name.toLowerCase().includes(q) || e.email.toLowerCase().includes(q)
      );
    }
    return [...list].sort((a, b) => {
      let cmp = 0;
      if (sortField === 'rank')        cmp = a.rank - b.rank;
      else if (sortField === 'name')   cmp = a.name.localeCompare(b.name);
      else if (sortField === 'trade_group') cmp = a.trade_group.localeCompare(b.trade_group);
      else if (sortField === 'trade_type') cmp = a.trade_type.localeCompare(b.trade_type);
      return sortDir === 'asc' ? cmp : -cmp;
    });
  }, [engineers, nameFilter, sortField, sortDir]);

  const SortIcon = ({ field }: { field: SortField }) => {
    if (sortField !== field) return <ChevronUp size={13} className="opacity-30" />;
    return sortDir === 'asc' ? <ChevronUp size={13} /> : <ChevronDown size={13} />;
  };

  const wetCount = engineers.filter((e) => e.trade_type === 'WET').length;
  const dryCount = engineers.filter((e) => e.trade_type === 'DRY').length;

  const thStyle = { color: colors.brand.blue, backgroundColor: colors.primary.subtle };
  const thClass = 'px-4 py-3 text-left text-xs font-bold uppercase tracking-wider cursor-pointer select-none';

  return (
     <div className="min-h-screen bg-gray-50" style={{ fontFamily: 'Montserrat, sans-serif' }}>

    {/* Banner Header */}
    <div
      style={{
        position: "relative",
        backgroundImage: "url(/aspectbackground.png)",
        backgroundSize: "cover",
        backgroundPosition: "center",
        height: "180px",
      }}
    >
      {/* Dark overlay */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          backgroundColor: "rgba(0,0,0,0.55)",
        }}
      />

      <div className="relative z-10 max-w-7xl mx-auto px-6 py-10">
        <h1 className="text-3xl font-bold text-white">
          Engineer Application AI Ranking
        </h1>
        <p className="text-white/80 text-sm mt-2">
          Browse and filter engineer applications by name or trade group
        </p>
      </div>
    </div>

    {/* Page Content */}
    <div className="p-6 max-w-7xl mx-auto space-y-6"></div>
      {/* Heading */}
      <div>
        <h2 className="text-2xl font-bold" style={{ color: colors.brand.blue }}>
          Chumley Engineer Application AI Ranking
        </h2>
        <p className="text-sm mt-1" style={{ color: colors.grayscale.subtle }}>
          Browse and filter engineer applications by name or trade group
        </p>
      </div>

      {/* Stats */}
      {!loading && !error && (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div className="bg-white rounded-xl border p-4 flex items-center gap-4"
            style={{ borderColor: colors.grayscale.border.default }}>
            <div className="rounded-full p-3" style={{ backgroundColor: colors.primary.subtle }}>
              <Users size={20} style={{ color: colors.brand.blue }} />
            </div>
            <div>
              <p className="text-xs font-semibold uppercase" style={{ color: colors.grayscale.subtle }}>Total Engineers</p>
              <p className="text-2xl font-bold" style={{ color: colors.brand.blue }}>{engineers.length}</p>
            </div>
          </div>

          <div className="bg-white rounded-xl border p-4 flex items-center gap-4"
            style={{ borderColor: colors.grayscale.border.default }}>
            <div className="rounded-full p-3" style={{ backgroundColor: '#dbeafe' }}>
              <Droplets size={20} style={{ color: '#1d4ed8' }} />
            </div>
            <div>
              <p className="text-xs font-semibold uppercase" style={{ color: colors.grayscale.subtle }}>WET Trade</p>
              <p className="text-2xl font-bold" style={{ color: '#1d4ed8' }}>{wetCount}</p>
            </div>
          </div>

          <div className="bg-white rounded-xl border p-4 flex items-center gap-4"
            style={{ borderColor: colors.grayscale.border.default }}>
            <div className="rounded-full p-3" style={{ backgroundColor: '#fef9c3' }}>
              <Zap size={20} style={{ color: '#854d0e' }} />
            </div>
            <div>
              <p className="text-xs font-semibold uppercase" style={{ color: colors.grayscale.subtle }}>DRY Trade</p>
              <p className="text-2xl font-bold" style={{ color: '#854d0e' }}>{dryCount}</p>
            </div>
          </div>
        </div>
      )}

      {/* Filters */}
      <div className="bg-white rounded-xl border p-4 flex flex-wrap gap-4 items-end"
        style={{ borderColor: colors.grayscale.border.default }}>
        <div className="flex-1 min-w-[200px]">
          <label className="block text-xs font-semibold mb-1" style={{ color: colors.grayscale.subtle }}>
            Search by Name / Email
          </label>
          <div className="relative">
            <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2" style={{ color: colors.grayscale.caption }} />
            <input
              type="text"
              placeholder="e.g. James Pegg"
              value={nameFilter}
              onChange={(e) => setNameFilter(e.target.value)}
              className="w-full pl-9 pr-3 py-2 border rounded-lg text-sm focus:outline-none focus:ring-2"
              style={{ borderColor: colors.grayscale.border.default, color: colors.grayscale.body }}
            />
          </div>
        </div>

        <div className="flex-1 min-w-[200px]">
          <label className="block text-xs font-semibold mb-1" style={{ color: colors.grayscale.subtle }}>
            Filter by Trade Group
          </label>
          <select
            value={tradeFilter}
            onChange={(e) => setTradeFilter(e.target.value)}
            className="w-full px-3 py-2 border rounded-lg text-sm focus:outline-none focus:ring-2"
            style={{ borderColor: colors.grayscale.border.default, color: colors.grayscale.body }}
          >
            <option value="all">All Trade Groups</option>
            {tradeGroups.map((t) => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </div>

        {(nameFilter || tradeFilter !== 'all') && (
          <button
            onClick={() => { setNameFilter(''); setTradeFilter('all'); }}
            className="px-3 py-2 text-xs rounded-lg border font-medium hover:bg-gray-50 transition"
            style={{ borderColor: colors.grayscale.border.default, color: colors.grayscale.subtle }}
          >
            Clear Filters
          </button>
        )}
      </div>

      {/* Table */}
      <div className="bg-white rounded-xl border overflow-hidden"
        style={{ borderColor: colors.grayscale.border.default }}>
        {loading ? (
          <div className="flex flex-col items-center justify-center py-20 gap-3">
            <div
              className="w-10 h-10 border-4 border-t-transparent rounded-full animate-spin"
              style={{ borderColor: colors.brand.blue, borderTopColor: 'transparent' }}
            />
            <p style={{ color: colors.grayscale.subtle }}>Loading engineer rankings...</p>
          </div>
        ) : error ? (
          <div className="flex flex-col items-center justify-center py-20 gap-3">
            <AlertCircle size={36} style={{ color: colors.support.red }} />
            <p className="font-semibold" style={{ color: colors.support.red }}>Failed to load rankings</p>
            <p className="text-sm text-center max-w-md" style={{ color: colors.grayscale.subtle }}>{error}</p>
          </div>
        ) : filtered.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-20 gap-3">
            <Users size={36} style={{ color: colors.grayscale.caption }} />
            <p className="font-semibold" style={{ color: colors.grayscale.subtle }}>No engineers found</p>
            <p className="text-sm" style={{ color: colors.grayscale.caption }}>Try adjusting your filters</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr style={{ borderBottom: `2px solid ${colors.grayscale.border.default}` }}>
                  {(
                    [
                      { field: 'rank' as SortField,        label: 'Rank' },
                      { field: 'name' as SortField,        label: 'Engineer Name' },
                      { field: 'trade_group' as SortField, label: 'Trade Group' },
                      { field: 'trade_type' as SortField,  label: 'Type' },
                    ] as { field: SortField; label: string }[]
                  ).map(({ field, label }) => (
                    <th key={field} className={thClass} style={thStyle} onClick={() => handleSort(field)}>
                      <span className="flex items-center gap-1">
                        {label} <SortIcon field={field} />
                      </span>
                    </th>
                  ))}
                  <th className={thClass} style={thStyle}>Email</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((eng, i) => (
                  <tr
                    key={eng.id || i}
                    className="transition-colors hover:bg-gray-50"
                    style={{ borderBottom: `1px solid ${colors.grayscale.border.disabled}` }}
                  >
                    <td className="px-4 py-3 w-16"><RankBadge rank={eng.rank} /></td>
                    <td className="px-4 py-3 font-medium" style={{ color: colors.grayscale.title }}>{eng.name}</td>
                    <td className="px-4 py-3" style={{ color: colors.grayscale.body }}>{eng.trade_group}</td>
                    <td className="px-4 py-3"><TradeTypeBadge type={eng.trade_type} /></td>
                    <td className="px-4 py-3 text-xs" style={{ color: colors.grayscale.caption }}>{eng.email || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>

            <div
              className="px-4 py-3 text-xs"
              style={{ color: colors.grayscale.caption, borderTop: `1px solid ${colors.grayscale.border.disabled}` }}
            >
              Showing {filtered.length} of {engineers.length} engineer{engineers.length !== 1 ? 's' : ''}
              {tradeFilter !== 'all' && ` in "${tradeFilter}"`}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default EngineerRankingPage;
