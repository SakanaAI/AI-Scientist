import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { CssBaseline, ThemeProvider as MuiThemeProvider } from '@mui/material';
import { LocalizationProvider } from '@mui/x-date-pickers';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import { useTheme } from './context/ThemeContext';
import { useAuth } from './context/AuthContext';

// Layout components
import MainLayout from './components/layout/MainLayout';
import AuthLayout from './components/layout/AuthLayout';

// Pages
import Dashboard from './pages/Dashboard';
import ExperimentsList from './pages/experiments/ExperimentsList';
import ExperimentDetail from './pages/experiments/ExperimentDetail';
import CreateExperiment from './pages/experiments/CreateExperiment';
import PapersList from './pages/papers/PapersList';
import PaperDetail from './pages/papers/PaperDetail';
import ResultsVisualization from './pages/results/ResultsVisualization';
import Login from './pages/auth/Login';
import Register from './pages/auth/Register';
import NotFound from './pages/NotFound';

// Protected route component
const ProtectedRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { isAuthenticated, loading } = useAuth();
  
  if (loading) {
    return <div>Loading...</div>;
  }
  
  return isAuthenticated ? <>{children}</> : <Navigate to="/auth/login" />;
};

const App: React.FC = () => {
  const { theme } = useTheme();
  
  return (
    <MuiThemeProvider theme={theme}>
      <LocalizationProvider dateAdapter={AdapterDateFns}>
        <CssBaseline />
        <Routes>
          {/* Auth routes */}
          <Route path="/auth" element={<AuthLayout />}>
            <Route path="login" element={<Login />} />
            <Route path="register" element={<Register />} />
          </Route>
          
          {/* Main app routes */}
          <Route path="/" element={
            <ProtectedRoute>
              <MainLayout />
            </ProtectedRoute>
          }>
            <Route index element={<Dashboard />} />
            <Route path="experiments">
              <Route index element={<ExperimentsList />} />
              <Route path="new" element={<CreateExperiment />} />
              <Route path=":id" element={<ExperimentDetail />} />
            </Route>
            <Route path="papers">
              <Route index element={<PapersList />} />
              <Route path=":id" element={<PaperDetail />} />
            </Route>
            <Route path="results" element={<ResultsVisualization />} />
          </Route>
          
          {/* 404 route */}
          <Route path="*" element={<NotFound />} />
        </Routes>
      </LocalizationProvider>
    </MuiThemeProvider>
  );
};

export default App; 