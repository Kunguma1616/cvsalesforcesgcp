import React, { useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { X, ChevronDown, FileText, Upload, Trophy } from 'lucide-react';
import { colors } from '../config/colors';

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ isOpen, onClose }) => {
  const location = useLocation();
  const [expandedSection, setExpandedSection] = useState<string | null>('CV_MANAGEMENT');

  const cvManagementItems = [
    { icon: FileText, label: 'CV Dashboard', href: '/dashboard' },
    { icon: Upload, label: 'Upload CV', href: '/upload' },
  ];

  const analysisItems = [
    { icon: Trophy, label: 'Engineer AI Ranking', href: '/engineer-ranking' },
  ];

  const toggleSection = (section: string) => {
    setExpandedSection(expandedSection === section ? null : section);
  };

  const isActive = (href: string) => location.pathname === href;

  return (
    <>
      {isOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden"
          onClick={onClose}
        />
      )}

      <aside
        className={`fixed lg:static top-0 left-0 w-64 h-screen transition-all duration-300 z-50 lg:z-0 bg-white border-r border-gray-200 flex flex-col ${
          isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'
        }`}
      >
        {/* Logo */}
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <img src="/aspectLogo.svg" alt="Aspect" className="h-9" />
            <button
              onClick={onClose}
              className="lg:hidden text-gray-500 hover:bg-gray-100 p-1 rounded"
            >
              <X size={20} />
            </button>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 overflow-y-auto p-4">
          {/* CV MANAGEMENT */}
          <div className="mb-4">
            <button
              onClick={() => toggleSection('CV_MANAGEMENT')}
              className="w-full flex items-center justify-between px-3 py-2 rounded-lg transition-all hover:bg-gray-50"
            >
              <span className="text-xs font-bold tracking-wider" style={{ color: colors.brand.blue }}>
                CV MANAGEMENT
              </span>
              <ChevronDown
                size={16}
                className={`transition-transform ${expandedSection === 'CV_MANAGEMENT' ? 'rotate-180' : ''}`}
                style={{ color: colors.brand.blue }}
              />
            </button>

            {expandedSection === 'CV_MANAGEMENT' && (
              <div className="mt-1 space-y-0.5">
                {cvManagementItems.map((item) => {
                  const Icon = item.icon;
                  const active = isActive(item.href);
                  return (
                    <NavLink
                      key={item.label}
                      to={item.href}
                      onClick={onClose}
                      className={`flex items-center gap-3 px-3 py-2 rounded-lg transition-all text-sm font-medium ${
                        active ? 'text-white' : 'text-gray-700 hover:bg-gray-100'
                      }`}
                      style={active ? { backgroundColor: colors.brand.blue } : {}}
                    >
                      <Icon size={17} style={{ color: active ? '#fff' : colors.brand.blue }} />
                      <span>{item.label}</span>
                    </NavLink>
                  );
                })}
              </div>
            )}
          </div>

          {/* ANALYSIS TOOLS */}
          <div className="mb-4">
            <button
              onClick={() => toggleSection('ANALYSIS')}
              className="w-full flex items-center justify-between px-3 py-2 rounded-lg transition-all hover:bg-gray-50"
            >
              <span className="text-xs font-bold tracking-wider" style={{ color: colors.brand.blue }}>
                ANALYSIS TOOLS
              </span>
              <ChevronDown
                size={16}
                className={`transition-transform ${expandedSection === 'ANALYSIS' ? 'rotate-180' : ''}`}
                style={{ color: colors.brand.blue }}
              />
            </button>

            {expandedSection === 'ANALYSIS' && (
              <div className="mt-1 space-y-0.5">
                {analysisItems.map((item) => {
                  const Icon = item.icon;
                  const active = isActive(item.href);
                  return (
                    <NavLink
                      key={item.label}
                      to={item.href}
                      onClick={onClose}
                      className={`flex items-center gap-3 px-3 py-2 rounded-lg transition-all text-sm font-medium ${
                        active ? 'text-white' : 'text-gray-700 hover:bg-gray-100'
                      }`}
                      style={active ? { backgroundColor: colors.brand.blue } : {}}
                    >
                      <Icon size={17} style={{ color: active ? '#fff' : colors.brand.blue }} />
                      <span>{item.label}</span>
                    </NavLink>
                  );
                })}
              </div>
            )}
          </div>
        </nav>

        {/* Bottom */}
        <div className="p-4 border-t border-gray-200">
        </div>
      </aside>
    </>
  );
};

export default Sidebar;
