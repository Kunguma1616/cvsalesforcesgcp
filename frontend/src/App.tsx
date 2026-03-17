import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './hooks/useAuth';
import { ProtectedRoute } from './components/ProtectedRoute';
import MainLayout from './components/MainLayout';
import LoginPage from './pages/LoginPage';
import HomePage from './pages/HomePage';
import UploadCVPage from './pages/UploadCVPage';
import AnalysisResultPage from './pages/AnalysisResultPage';
import AnalysisReportsPage from './pages/AnalysisReportsPage';
import EngineerRankingPage from './pages/EngineerRankingPage';

function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          {/* Default: redirect to login */}
          <Route path="/" element={<Navigate to="/login" replace />} />

          {/* Public Routes */}
          <Route path="/login" element={<LoginPage />} />

          {/* Protected Routes */}
          <Route
            path="/dashboard"
            element={
              <ProtectedRoute>
                <MainLayout>
                  <HomePage />
                </MainLayout>
              </ProtectedRoute>
            }
          />
          <Route
            path="/upload"
            element={
              <ProtectedRoute>
                <MainLayout>
                  <UploadCVPage />
                </MainLayout>
              </ProtectedRoute>
            }
          />
          <Route
            path="/analysis-result"
            element={
              <ProtectedRoute>
                <MainLayout>
                  <AnalysisResultPage />
                </MainLayout>
              </ProtectedRoute>
            }
          />
          <Route
            path="/reports"
            element={
              <ProtectedRoute>
                <MainLayout>
                  <AnalysisReportsPage />
                </MainLayout>
              </ProtectedRoute>
            }
          />
          <Route
            path="/job-analysis"
            element={
              <ProtectedRoute>
                <MainLayout>
                  <AnalysisReportsPage />
                </MainLayout>
              </ProtectedRoute>
            }
          />
          <Route
            path="/skill-reports"
            element={
              <ProtectedRoute>
                <MainLayout>
                  <AnalysisReportsPage />
                </MainLayout>
              </ProtectedRoute>
            }
          />
          <Route
            path="/bulk-processing"
            element={
              <ProtectedRoute>
                <MainLayout>
                  <UploadCVPage />
                </MainLayout>
              </ProtectedRoute>
            }
          />
          <Route
            path="/engineer-ranking"
            element={
              <ProtectedRoute>
                <MainLayout>
                  <EngineerRankingPage />
                </MainLayout>
              </ProtectedRoute>
            }
          />

          {/* Fallback */}
          <Route path="*" element={<Navigate to="/login" replace />} />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  );
}

export default App;
