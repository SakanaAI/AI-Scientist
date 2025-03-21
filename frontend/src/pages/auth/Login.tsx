import React, { useState } from 'react';
import { useNavigate, Link as RouterLink } from 'react-router-dom';
import {
  Box,
  TextField,
  Button,
  Typography,
  Link,
  Alert,
  InputAdornment,
  IconButton,
  CircularProgress,
  Divider,
  Paper
} from '@mui/material';
import { Visibility, VisibilityOff, Login as LoginIcon } from '@mui/icons-material';
import { useAuth } from '../../context/AuthContext';
import { useTheme } from '../../context/ThemeContext';

const Login: React.FC = () => {
  const navigate = useNavigate();
  const { login } = useAuth();
  const { mode } = useTheme();
  
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!email || !password) {
      setError('Please enter both email and password');
      return;
    }
    
    try {
      setError(null);
      setLoading(true);
      await login(email, password);
      navigate('/');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed. Please check your credentials.');
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <Box component="form" onSubmit={handleSubmit} noValidate>
      <Typography variant="h5" fontWeight="600" textAlign="center" gutterBottom>
        Sign In
      </Typography>
      
      <Typography variant="body2" color="text.secondary" align="center" sx={{ mb: 3 }}>
        Enter your credentials to continue
      </Typography>
      
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}
      
      <TextField
        margin="normal"
        required
        fullWidth
        id="email"
        label="Email Address"
        name="email"
        autoComplete="email"
        autoFocus
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        disabled={loading}
        variant="outlined"
        sx={{ mb: 2 }}
      />
      
      <TextField
        margin="normal"
        required
        fullWidth
        name="password"
        label="Password"
        type={showPassword ? 'text' : 'password'}
        id="password"
        autoComplete="current-password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        disabled={loading}
        variant="outlined"
        sx={{ mb: 1 }}
        InputProps={{
          endAdornment: (
            <InputAdornment position="end">
              <IconButton
                aria-label="toggle password visibility"
                onClick={() => setShowPassword(!showPassword)}
                edge="end"
              >
                {showPassword ? <VisibilityOff /> : <Visibility />}
              </IconButton>
            </InputAdornment>
          )
        }}
      />
      
      <Box sx={{ mt: 1, textAlign: 'right', mb: 3 }}>
        <Link component={RouterLink} to="/auth/forgot-password" variant="body2" color="primary">
          Forgot password?
        </Link>
      </Box>
      
      <Button
        type="submit"
        fullWidth
        variant="contained"
        sx={{ 
          py: 1.5, 
          fontSize: '1rem',
          fontWeight: 500,
          mb: 3
        }}
        disabled={loading}
        startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <LoginIcon />}
      >
        {loading ? 'Signing in...' : 'Sign In'}
      </Button>
      
      <Divider sx={{ mb: 3 }}>
        <Typography variant="body2" color="text.secondary">
          OR
        </Typography>
      </Divider>
      
      <Box sx={{ textAlign: 'center', mb: 2 }}>
        <Typography variant="body2" color="text.secondary">
          Don't have an account?{' '}
          <Link component={RouterLink} to="/auth/register" variant="body2" sx={{ fontWeight: 500 }}>
            Sign Up
          </Link>
        </Typography>
      </Box>
      
      <Paper 
        elevation={0} 
        sx={{ 
          p: 2, 
          mt: 3, 
          bgcolor: mode === 'dark' ? 'rgba(0, 0, 0, 0.1)' : 'rgba(0, 0, 0, 0.02)',
          border: '1px solid',
          borderColor: mode === 'dark' ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.05)',
          borderRadius: 2
        }}
      >
        <Typography variant="body2" color="text.secondary" align="center" fontWeight={500}>
          Demo credentials:
        </Typography>
        <Typography variant="body2" color="text.secondary" align="center">
          Email: <strong>demo@example.com</strong>, Password: <strong>password</strong>
        </Typography>
      </Paper>
    </Box>
  );
};

export default Login; 