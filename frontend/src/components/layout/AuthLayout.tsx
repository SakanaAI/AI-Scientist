import React from 'react';
import { Outlet } from 'react-router-dom';
import { 
  Container, 
  Box, 
  Paper, 
  Typography,
  IconButton,
  useTheme as useMuiTheme
} from '@mui/material';
import { LightMode, DarkMode } from '@mui/icons-material';
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
        background: 'linear-gradient(135deg, #3f51b5 0%, #00bcd4 100%)',
        p: 2
      }}
    >
      <Box 
        sx={{ 
          position: 'absolute', 
          top: 16, 
          right: 16 
        }}
      >
        <IconButton 
          onClick={toggleTheme} 
          sx={{ 
            color: 'white',
            bgcolor: 'rgba(255, 255, 255, 0.1)',
            '&:hover': {
              bgcolor: 'rgba(255, 255, 255, 0.2)',
            }
          }}
        >
          {mode === 'light' ? <DarkMode /> : <LightMode />}
        </IconButton>
      </Box>
      
      <Container maxWidth="sm">
        <Paper
          elevation={6}
          sx={{
            p: 4,
            borderRadius: 2,
            boxShadow: 'rgba(0, 0, 0, 0.1) 0px 10px 50px',
            overflow: 'hidden',
          }}
        >
          <Box 
            sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              mb: 4
            }}
          >
            <Typography 
              component="h1" 
              variant="h4" 
              sx={{ 
                fontWeight: 700,
                mb: 1,
                color: 'primary.main'
              }}
            >
              AI-Scientist
            </Typography>
            <Typography 
              variant="body1" 
              color="text.secondary"
              align="center"
            >
              Automated Scientific Discovery Platform
            </Typography>
          </Box>
          
          <Outlet />
        </Paper>
      </Container>
      
      <Typography 
        variant="body2" 
        color="white" 
        align="center"
        sx={{ mt: 4, opacity: 0.8 }}
      >
        © {new Date().getFullYear()} AI-Scientist Platform
      </Typography>
    </Box>
  );
};

export default AuthLayout; 