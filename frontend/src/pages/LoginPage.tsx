import React, { useEffect, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { AlertCircle } from 'lucide-react';
import { colors } from '../config/colors';
import { useAuth } from '../hooks/useAuth';

const LoginPage: React.FC = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const { login } = useAuth();

  useEffect(() => {
    const sessionToken = searchParams.get('session');
    const userEmail = searchParams.get('email');
    const userName = searchParams.get('user');
    const errorParam = searchParams.get('error');

    if (errorParam) {
      handleLoginError(errorParam);
    } else if (sessionToken && userEmail) {
      login(sessionToken, userEmail, userName || 'User');
      navigate('/dashboard');
    }
  }, [searchParams, navigate]);

  const handleMicrosoftLogin = () => {
    setLoading(true);
    setError('');
    window.location.href = '/api/auth/microsoft';
  };

  const handleLoginError = (errorCode: string) => {
    const errorMessages: Record<string, string> = {
      oauth_error: 'OAuth login was cancelled or failed.',
      no_code: 'No authorization code received from Microsoft.',
      token_exchange_failed: 'Failed to exchange code for token.',
      no_token: 'No access token received.',
      user_info_failed: 'Failed to retrieve user information.',
      no_email: 'Email not found in user profile.',
      unauthorized_domain: 'Your email domain is not authorized to access this application.',
      server_error: 'Server error occurred during login.'
    };
    setError(errorMessages[errorCode] || 'An error occurred during login.');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Card */}
        <div className="bg-white rounded-2xl shadow-lg p-8">
          {/* Logo */}
          <div className="flex justify-center mb-8">
            <img src="/aspectLogo.svg" alt="Aspect" className="h-12" />
          </div>

          {/* Title */}
          <h1 className="text-3xl font-bold text-center mb-2" style={{ color: colors.brand.blue }}>
            "Recruitment Portal"
          </h1>
          <p className="text-center text-gray-600 text-sm mb-8">
            Sign in with your Microsoft account to access the platform
          </p>

          {/* Error Message */}
          {error && (
            <div className="mb-6 flex items-start gap-3 bg-red-50 border border-red-200 rounded-lg p-4">
              <AlertCircle size={18} className="text-red-600 mt-0.5 shrink-0" />
              <div>
                <p className="text-red-600 text-sm font-medium">Login Failed</p>
                <p className="text-red-600 text-sm mt-1">{error}</p>
              </div>
            </div>
          )}

          {/* Microsoft Login Button */}
          <button
            onClick={handleMicrosoftLogin}
            disabled={loading}
            className="w-full py-3 rounded-lg font-semibold text-white text-sm transition-all hover:opacity-90 disabled:opacity-60 disabled:cursor-not-allowed flex items-center justify-center gap-3 mb-6"
            style={{ backgroundColor: colors.brand.blue }}
          >
            {loading ? (
              <>
                <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                </svg>
                Signing in...
              </>
            ) : (
              <>
                <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M11.4 24h-8.98C1.05 24 0 22.95 0 21.58V2.42C0 1.05 1.04 0 2.42 0h19.16C22.96 0 24 1.05 24 2.42v13.84h-2.5V2.42c0-.77-.77-1.5-1.58-1.5H2.42c-.8 0-1.42.75-1.42 1.5v19.16c0 .77.62 1.42 1.42 1.42H11.4V24z" />
                  <path d="M17.38 19.33H5.5v-2.5h11.88v2.5zm0-4.17H5.5v-2.5h11.88v2.5zm0-4.16H5.5v-2.5h11.88v2.5z" />
                </svg>
                Sign in with Microsoft
              </>
            )}
          </button>

          {/* Help Text */}
          <p className="text-center text-gray-500 text-xs">
            You need a valid company email address to access this platform
          </p>
        </div>

        {/* Footer */}
        <p className="text-center text-gray-600 text-xs mt-8">
          © 2024 Aspect Services. All rights reserved.
        </p>
      </div>
    </div>
  );
};

export default LoginPage;
