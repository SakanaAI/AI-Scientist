import React from 'react';
import { Outlet } from 'react-router-dom';
import { 
  Container, 
  Box, 
  Paper, 
  Typography,
  IconButton,
  useTheme as useMuiTheme,
  alpha
} from '@mui/material';
import { LightMode, DarkMode, Science } from '@mui/icons-material';
import { useTheme } from '../../context/ThemeContext';

const AuthLayout: React.FC = () => {
  const { toggleTheme, mode } = useTheme();
  const muiTheme = useMuiTheme();
  
  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        background: mode === 'dark'
          ? 'linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%)'
          : 'linear-gradient(135deg, #3f51b5 0%, #00bcd4 100%)',
        p: 2,
        position: 'relative',
        overflow: 'hidden',
        '&::before': {
          content: '""',
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: mode === 'dark'
            ? 'url("data:image/svg+xml,%3Csvg width=\'100\' height=\'100\' viewBox=\'0 0 100 100\' xmlns=\'http://www.w3.org/2000/svg\'%3E%3Cpath d=\'M11 18c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm48 25c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm-43-7c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm63 31c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM34 90c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm56-76c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM12 86c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm28-65c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm23-11c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-6 60c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm29 22c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zM32 63c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm57-13c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-9-21c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM60 91c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM35 41c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM12 60c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2z\' fill=\'%232563eb\' fill-opacity=\'0.05\' fill-rule=\'evenodd\'/%3E%3C/svg%3E")'
            : 'url("data:image/svg+xml,%3Csvg width=\'100\' height=\'100\' viewBox=\'0 0 100 100\' xmlns=\'http://www.w3.org/2000/svg\'%3E%3Cpath d=\'M11 18c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm48 25c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm-43-7c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm63 31c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM34 90c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm56-76c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM12 86c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm28-65c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm23-11c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-6 60c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm29 22c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zM32 63c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm57-13c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-9-21c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM60 91c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM35 41c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM12 60c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2z\' fill=\'%23e0f2fe\' fill-opacity=\'0.1\' fill-rule=\'evenodd\'/%3E%3C/svg%3E")',
          opacity: 0.8
        }
      }}
    >
      <Box 
        sx={{ 
          position: 'absolute', 
          top: 16, 
          right: 16,
          zIndex: 2
        }}
      >
        <IconButton 
          onClick={toggleTheme} 
          sx={{ 
            color: 'white',
            bgcolor: 'rgba(255, 255, 255, 0.1)',
            backdropFilter: 'blur(8px)',
            '&:hover': {
              bgcolor: 'rgba(255, 255, 255, 0.2)',
            }
          }}
        >
          {mode === 'light' ? <DarkMode /> : <LightMode />}
        </IconButton>
      </Box>
      
      <Container maxWidth="sm" sx={{ position: 'relative', zIndex: 1 }}>
        <Box 
          sx={{ 
            display: 'flex', 
            flexDirection: 'column',
            alignItems: 'center',
            mb: 5
          }}
        >
          <Box 
            sx={{ 
              display: 'flex',
              alignItems: 'center',
              gap: 1.5,
              mb: 1
            }}
          >
            <Science sx={{ fontSize: 40, color: 'white' }} />
            <Typography 
              variant="h3" 
              component="h1" 
              sx={{ 
                fontWeight: 700,
                background: 'linear-gradient(45deg, #ffffff 30%, #bbdefb 90%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                textShadow: '0px 2px 5px rgba(0,0,0,0.2)'
              }}
            >
              AI-Scientist
            </Typography>
          </Box>
          <Typography 
            variant="h6" 
            align="center"
            sx={{ 
              color: 'white',
              opacity: 0.9,
              fontWeight: 400,
              textShadow: '0px 1px 3px rgba(0,0,0,0.2)',
              maxWidth: '80%'
            }}
          >
            Automated Scientific Discovery Platform
          </Typography>
        </Box>

        <Paper
          elevation={mode === 'dark' ? 5 : 2}
          sx={{
            p: 4,
            borderRadius: 3,
            boxShadow: mode === 'dark' 
              ? '0 10px 30px rgba(0, 0, 0, 0.5)' 
              : '0 10px 30px rgba(0, 0, 0, 0.1)',
            backdropFilter: 'blur(12px)',
            background: mode === 'dark'
              ? alpha(muiTheme.palette.background.paper, 0.8)
              : alpha(muiTheme.palette.background.paper, 0.8),
            border: mode === 'dark' 
              ? '1px solid rgba(255, 255, 255, 0.08)'
              : '1px solid rgba(255, 255, 255, 0.1)',
          }}
        >
          <Outlet />
        </Paper>
      </Container>
      
      <Typography 
        variant="body2" 
        color="white" 
        align="center"
        sx={{ mt: 4, opacity: 0.7 }}
      >
        © {new Date().getFullYear()} AI-Scientist Platform
      </Typography>
    </Box>
  );
};

export default AuthLayout; 