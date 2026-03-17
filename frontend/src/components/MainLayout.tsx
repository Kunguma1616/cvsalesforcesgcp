import React, { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Menu, Bell, User, LogOut } from 'lucide-react';
import Sidebar from './Sidebar';
import { colors } from '../config/colors';
import { useAuth } from '../hooks/useAuth';

interface MainLayoutProps {
  children: React.ReactNode;
}

const PAGE_TITLES: Record<string, string> = {
  '/': 'CV Analysis Dashboard',
  '/upload': 'Upload CV',
  '/reports': 'Analysis Reports',
  '/analysis-result': 'Analysis Result',
  '/job-analysis': 'Job Analysis',
  '/skill-reports': 'Skill Reports',
  '/bulk-processing': 'Bulk Processing',
  '/engineer-ranking': 'Engineer AI Ranking',
};

export const MainLayout: React.FC<MainLayoutProps> = ({ children }) => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();
  const { user, logout } = useAuth();

  const pageTitle = PAGE_TITLES[location.pathname] ?? 'CV Parser';

  const handleLogout = async () => {
    await logout();
    navigate('/login');
  };

  return (
    <div className="flex h-screen bg-gray-50">
      <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />

      <div className="flex-1 flex flex-col overflow-hidden">
        <header
          className="h-16 border-b shadow-sm flex items-center justify-between px-6"
          style={{ backgroundColor: 'white', borderColor: colors.grayscale.border.default }}
        >
          <div className="flex items-center gap-4">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="lg:hidden p-2 hover:bg-gray-100 rounded-lg transition-all"
              style={{ color: colors.brand.blue }}
            >
              <Menu size={24} />
            </button>
            <h1 className="text-xl font-bold hidden sm:block" style={{ color: colors.brand.blue }}>
              {pageTitle}
            </h1>
          </div>

          <div className="flex items-center gap-4 relative">
            <button className="p-2 hover:bg-gray-100 rounded-lg transition-all" style={{ color: colors.grayscale.subtle }}>
              <Bell size={20} />
            </button>
            <div className="relative">
              <button 
                onClick={() => setUserMenuOpen(!userMenuOpen)}
                className="flex items-center gap-2 px-3 py-1 hover:bg-gray-100 rounded-lg transition-all"
                style={{ color: colors.grayscale.subtle }}
              >
                <User size={20} />
                <span className="text-sm font-medium text-gray-700 hidden sm:inline">
                  {user?.name || user?.email}
                </span>
              </button>

              {/* User Dropdown Menu */}
              {userMenuOpen && (
                <div className="absolute right-0 mt-2 w-48 bg-white rounded-lg shadow-lg border border-gray-200 z-50">
                  <div className="p-4 border-b border-gray-200">
                    <p className="text-sm font-semibold text-gray-700">{user?.name || 'User'}</p>
                    <p className="text-xs text-gray-500 mt-1">{user?.email}</p>
                  </div>
                  <button
                    onClick={() => {
                      setUserMenuOpen(false);
                      handleLogout();
                    }}
                    className="w-full flex items-center gap-2 px-4 py-2 text-red-600 hover:bg-red-50 transition-all text-sm font-medium"
                  >
                    <LogOut size={16} />
                    Logout
                  </button>
                </div>
              )}
            </div>
          </div>
        </header>

        <main className="flex-1 overflow-auto">
          {children}
        </main>
      </div>
    </div>
  );
};

export default MainLayout;
