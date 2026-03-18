import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

interface User {
  name: string;
  email: string;
}

interface AuthContextType {
  isAuthenticated: boolean;
  user: User | null;
  sessionId: string | null;
  login: (sessionId: string, email: string, name: string) => void;
  logout: () => void;
  isLoading: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

// ✅ FIXED: fallback is now the production URL, not localhost
const API_URL = import.meta.env.VITE_API_URL || 'https://cv-parser-service-726237234326.europe-west2.run.app';

export const AuthProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [user, setUser] = useState<User | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Check if user has a valid session on mount
    const storedSessionId = localStorage.getItem('session_id');
    const storedEmail = localStorage.getItem('user_email');
    const storedName = localStorage.getItem('user_name');

    if (storedSessionId && storedEmail) {
      // Verify session with backend
      verifySession(storedSessionId, storedEmail, storedName);
    } else {
      setIsLoading(false);
    }
  }, []);

  const verifySession = async (sessionId: string, email: string, name: string | null) => {
    try {
      const response = await fetch(`${API_URL}/api/auth/verify/${sessionId}`);

      if (response.ok) {
        setSessionId(sessionId);
        setUser({
          email: email,
          name: name || 'User'
        });
        setIsAuthenticated(true);
      } else {
        // Session invalid
        localStorage.removeItem('session_id');
        localStorage.removeItem('user_email');
        localStorage.removeItem('user_name');
        setIsAuthenticated(false);
      }
    } catch (error) {
      console.error('Failed to verify session:', error);
      setIsAuthenticated(false);
    } finally {
      setIsLoading(false);
    }
  };

  const login = (sessionId: string, email: string, name: string) => {
    localStorage.setItem('session_id', sessionId);
    localStorage.setItem('user_email', email);
    localStorage.setItem('user_name', name);
    setSessionId(sessionId);
    setUser({ email, name });
    setIsAuthenticated(true);
  };

  const logout = async () => {
    const storedSessionId = localStorage.getItem('session_id');

    if (storedSessionId) {
      try {
        await fetch(`${API_URL}/api/auth/logout/${storedSessionId}`, {
          method: 'POST'
        });
      } catch (error) {
        console.error('Failed to logout:', error);
      }

      localStorage.removeItem('session_id');
      localStorage.removeItem('user_email');
      localStorage.removeItem('user_name');
    }

    setIsAuthenticated(false);
    setUser(null);
    setSessionId(null);
  };

  const value: AuthContextType = {
    isAuthenticated,
    user,
    sessionId,
    login,
    logout,
    isLoading
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
