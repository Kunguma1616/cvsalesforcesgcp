import { useState, useEffect } from 'react';

export interface TradeBreakdown {
  primary_trade: string;
  count: number;
}

export interface DashboardStats {
  total_applications: number;
  total_wet_trade: number;
  total_dry_trade: number;
  wet_trade_breakdown: TradeBreakdown[];
  dry_trade_breakdown: TradeBreakdown[];
}

const API_BASE_URL = import.meta.env.VITE_API_URL || '';

export const useDashboardStats = () => {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        setLoading(true);
        setError(null);
        const response = await fetch(`/api/dashboad/stats`);
        
        if (!response.ok) {
          throw new Error(`API Error: ${response.status}`);
        }
        
        const data = await response.json();
        setStats(data);
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Failed to fetch dashboard stats';
        setError(errorMessage);
        console.error('Dashboard stats fetch error:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchStats();
  }, []);

  return { stats, loading, error };
};
