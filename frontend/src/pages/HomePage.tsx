import React from 'react';
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from 'recharts';
import { colors } from '../config/colors';
import { useDashboardStats } from '../hooks/useDashboardStats';

export const HomePage: React.FC = () => {
  const { stats: dashboardData, loading, error } = useDashboardStats();

  // Color palette for pie charts - using brand colors
  const COLORS = [
    colors.primary.default,
    colors.primary.light,
    colors.warning.default,
    colors.warning.light,
    colors.support.green,
    colors.support.orange,
    colors.support.red,
    colors.grayscale.subtle,
  ];

  // Transform breakdown data for pie chart
  const wetTradeData = dashboardData?.wet_trade_breakdown.map(item => ({
    name: item.primary_trade,
    value: item.count
  })) || [];

  const dryTradeData = dashboardData?.dry_trade_breakdown.map(item => ({
    name: item.primary_trade,
    value: item.count
  })) || [];

  return (
    <div style={{ backgroundColor: colors.grayscale.negative }} className="min-h-screen">
      {/* Header Banner with Background Image */}
      <div
        style={{
          position: 'relative',
          backgroundImage: 'url(/aspectbackground.png)',
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          borderBottom: `1px solid ${colors.grayscale.border.default}`,
          overflow: 'hidden',
        }}
        className="sticky top-0 z-50"
      >
        {/* Dark overlay for text readability */}
        <div
          style={{
            position: 'absolute',
            inset: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.52)',
          }}
        />
        <div className="max-w-7xl mx-auto px-8 py-8" style={{ position: 'relative', zIndex: 1 }}>
          <h1 style={{ color: 'white', fontFamily: 'Montserrat', fontWeight: 800, textShadow: '0 1px 4px rgba(0,0,0,0.5)' }} className="text-4xl">
            Chumely Engineer Applications Dashboard
          </h1>
          <p style={{ color: 'rgba(255,255,255,0.85)', fontFamily: 'Montserrat', fontWeight: 400, textShadow: '0 1px 3px rgba(0,0,0,0.4)' }} className="mt-2 text-base">
            Real-time analytics & trade distribution insights
          </p>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-8 py-12">
        {/* Loading State */}
        {loading && (
          <div style={{ backgroundColor: colors.primary.subtle, borderLeft: `4px solid ${colors.primary.default}` }} className="rounded-lg p-6 text-center">
            <p style={{ color: colors.primary.default, fontFamily: 'Montserrat', fontWeight: 600 }} className="text-lg">
              Loading dashboard data...
            </p>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div style={{ backgroundColor: colors.error.subtle, borderLeft: `4px solid ${colors.error.default}` }} className="rounded-lg p-6">
            <p style={{ color: colors.error.default, fontFamily: 'Montserrat', fontWeight: 600 }} className="text-lg">
              Connection Error
            </p>
            <p style={{ color: colors.grayscale.body, fontFamily: 'Montserrat' }} className="text-sm mt-2">
              {error}
            </p>
          </div>
        )}

        {/* Dashboard Content */}
        {!loading && dashboardData && (
          <>
            {/* Key Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
              {[
                {
                  label: 'Total Applications',
                  value: dashboardData.total_applications.toLocaleString(),
                  subtitle: 'WET & DRY combined',
                  color: colors.primary.default,
                },
                {
                  label: 'WET Trade',
                  value: dashboardData.total_wet_trade.toLocaleString(),
                  subtitle: 'Active applications',
                  color: colors.warning.default,
                },
                {
                  label: 'DRY Trade',
                  value: dashboardData.total_dry_trade.toLocaleString(),
                  subtitle: 'Active applications',
                  color: colors.support.green,
                },
              ].map((metric, idx) => (
                <div key={idx} style={{ backgroundColor: 'white', borderTop: `3px solid ${metric.color}` }} className="rounded-lg p-6 shadow-sm hover:shadow-md transition-shadow">
                  <p style={{ color: colors.grayscale.subtle, fontFamily: 'Montserrat', fontWeight: 500 }} className="text-xs uppercase tracking-wider">
                    {metric.label}
                  </p>
                  <p style={{ color: metric.color, fontFamily: 'Montserrat', fontWeight: 800 }} className="text-4xl mt-2">
                    {metric.value}
                  </p>
                  <p style={{ color: colors.grayscale.subtle, fontFamily: 'Montserrat', fontWeight: 400 }} className="text-sm mt-2">
                    {metric.subtitle}
                  </p>
                </div>
              ))}
            </div>

            {/* Charts Section */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
              {/* WET Trade Distribution */}
              <div style={{ backgroundColor: 'white' }} className="rounded-lg shadow-sm p-8">
                <h2 style={{ color: colors.primary.default, fontFamily: 'Montserrat', fontWeight: 700 }} className="text-xl mb-6">
                  WET Trade Distribution
                </h2>
                {wetTradeData.length > 0 ? (
                  <>
                    <div style={{ width: '100%', height: 280 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart margin={{ top: 0, right: 0, bottom: 0, left: 0 }}>
                          <Pie
                            data={wetTradeData}
                            cx="45%"
                            cy="50%"
                            innerRadius={0}
                            outerRadius={85}
                            fill={colors.primary.default}
                            dataKey="value"
                            label={false}
                          >
                            {wetTradeData.map((entry, i) => (
                              <Cell key={`cell-${i}`} fill={COLORS[i % COLORS.length]} />
                            ))}
                          </Pie>
                          <Tooltip 
                            formatter={(value) => `${value} applications`}
                            contentStyle={{ backgroundColor: 'white', border: `2px solid ${colors.primary.default}`, fontFamily: 'Montserrat' }}
                          />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>
                    <div className="mt-6 space-y-2">
                      {wetTradeData.map((item, idx) => {
                        const total = wetTradeData.reduce((sum, i) => sum + i.value, 0);
                        const pct = ((item.value / total) * 100).toFixed(1);
                        return (
                          <div key={idx} className="flex items-center justify-between">
                            <div className="flex items-center gap-3">
                              <div 
                                className="w-3 h-3 rounded-full"
                                style={{ backgroundColor: COLORS[idx % COLORS.length] }}
                              ></div>
                              <p style={{ color: colors.grayscale.body, fontFamily: 'Montserrat', fontWeight: 500 }} className="text-sm">
                                {item.name}
                              </p>
                            </div>
                            <p style={{ color: colors.primary.default, fontFamily: 'Montserrat', fontWeight: 700 }} className="text-sm">
                              {item.value} ({pct}%)
                            </p>
                          </div>
                        );
                      })}
                    </div>
                  </>
                ) : (
                  <p style={{ color: colors.grayscale.subtle }} className="text-center py-8">No data available</p>
                )}
              </div>

              {/* DRY Trade Distribution */}
              <div style={{ backgroundColor: 'white' }} className="rounded-lg shadow-sm p-8">
                <h2 style={{ color: colors.primary.default, fontFamily: 'Montserrat', fontWeight: 700 }} className="text-xl mb-6">
                  DRY Trade Distribution
                </h2>
                {dryTradeData.length > 0 ? (
                  <>
                    <div style={{ width: '100%', height: 280 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart margin={{ top: 0, right: 0, bottom: 0, left: 0 }}>
                          <Pie
                            data={dryTradeData}
                            cx="45%"
                            cy="50%"
                            innerRadius={0}
                            outerRadius={85}
                            fill={colors.support.green}
                            dataKey="value"
                            label={false}
                          >
                            {dryTradeData.map((entry, i) => (
                              <Cell key={`cell-${i}`} fill={COLORS[i % COLORS.length]} />
                            ))}
                          </Pie>
                          <Tooltip 
                            formatter={(value) => `${value} applications`}
                            contentStyle={{ backgroundColor: 'white', border: `2px solid ${colors.primary.default}`, fontFamily: 'Montserrat' }}
                          />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>
                    <div className="mt-6 space-y-2">
                      {dryTradeData.map((item, idx) => {
                        const total = dryTradeData.reduce((sum, i) => sum + i.value, 0);
                        const pct = ((item.value / total) * 100).toFixed(1);
                        return (
                          <div key={idx} className="flex items-center justify-between">
                            <div className="flex items-center gap-3">
                              <div 
                                className="w-3 h-3 rounded-full"
                                style={{ backgroundColor: COLORS[idx % COLORS.length] }}
                              ></div>
                              <p style={{ color: colors.grayscale.body, fontFamily: 'Montserrat', fontWeight: 500 }} className="text-sm">
                                {item.name}
                              </p>
                            </div>
                            <p style={{ color: colors.primary.default, fontFamily: 'Montserrat', fontWeight: 700 }} className="text-sm">
                              {item.value} ({pct}%)
                            </p>
                          </div>
                        );
                      })}
                    </div>
                  </>
                ) : (
                  <p style={{ color: colors.grayscale.subtle }} className="text-center py-8">No data available</p>
                )}
              </div>
            </div>

            {/* Detailed Tables */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* WET Trade Table */}
              <div style={{ backgroundColor: 'white' }} className="rounded-lg shadow-sm p-8">
                <h2 style={{ color: colors.primary.default, fontFamily: 'Montserrat', fontWeight: 700 }} className="text-xl mb-6">
                  WET Trade Breakdown
                </h2>
                <div className="overflow-x-auto">
                  <table className="w-full" style={{ fontFamily: 'Montserrat' }}>
                    <thead>
                      <tr style={{ borderBottom: `2px solid ${colors.grayscale.border.default}` }}>
                        <th style={{ color: colors.grayscale.body, fontWeight: 600 }} className="text-left py-3 px-0 text-sm uppercase">Trade</th>
                        <th style={{ color: colors.grayscale.body, fontWeight: 600 }} className="text-right py-3 px-0 text-sm uppercase">Count</th>
                        <th style={{ color: colors.grayscale.body, fontWeight: 600 }} className="text-right py-3 px-0 text-sm uppercase">Share</th>
                      </tr>
                    </thead>
                    <tbody>
                      {wetTradeData.map((item, idx) => {
                        const total = wetTradeData.reduce((sum, i) => sum + i.value, 0);
                        const pct = ((item.value / total) * 100).toFixed(1);
                        return (
                          <tr key={idx} style={{ borderBottom: `1px solid ${colors.grayscale.border.disabled}` }} className="hover:bg-gray-50">
                            <td style={{ color: colors.grayscale.body, fontWeight: 500 }} className="py-3 px-0 text-sm">{item.name}</td>
                            <td style={{ color: colors.primary.default, fontWeight: 600 }} className="py-3 px-0 text-right text-sm">{item.value}</td>
                            <td style={{ color: colors.grayscale.subtle, fontWeight: 500 }} className="py-3 px-0 text-right text-sm">{pct}%</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* DRY Trade Table */}
              <div style={{ backgroundColor: 'white' }} className="rounded-lg shadow-sm p-8">
                <h2 style={{ color: colors.primary.default, fontFamily: 'Montserrat', fontWeight: 700 }} className="text-xl mb-6">
                  DRY Trade Breakdown
                </h2>
                <div className="overflow-x-auto">
                  <table className="w-full" style={{ fontFamily: 'Montserrat' }}>
                    <thead>
                      <tr style={{ borderBottom: `2px solid ${colors.grayscale.border.default}` }}>
                        <th style={{ color: colors.grayscale.body, fontWeight: 600 }} className="text-left py-3 px-0 text-sm uppercase">Trade</th>
                        <th style={{ color: colors.grayscale.body, fontWeight: 600 }} className="text-right py-3 px-0 text-sm uppercase">Count</th>
                        <th style={{ color: colors.grayscale.body, fontWeight: 600 }} className="text-right py-3 px-0 text-sm uppercase">Share</th>
                      </tr>
                    </thead>
                    <tbody>
                      {dryTradeData.map((item, idx) => {
                        const total = dryTradeData.reduce((sum, i) => sum + i.value, 0);
                        const pct = ((item.value / total) * 100).toFixed(1);
                        return (
                          <tr key={idx} style={{ borderBottom: `1px solid ${colors.grayscale.border.disabled}` }} className="hover:bg-gray-50">
                            <td style={{ color: colors.grayscale.body, fontWeight: 500 }} className="py-3 px-0 text-sm">{item.name}</td>
                            <td style={{ color: colors.primary.default, fontWeight: 600 }} className="py-3 px-0 text-right text-sm">{item.value}</td>
                            <td style={{ color: colors.grayscale.subtle, fontWeight: 500 }} className="py-3 px-0 text-right text-sm">{pct}%</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default HomePage;

