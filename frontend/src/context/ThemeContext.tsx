import React, { createContext, useContext, useState, useEffect } from 'react';
import { createTheme, Theme, PaletteMode } from '@mui/material';

interface ThemeContextType {
  mode: PaletteMode;
  toggleTheme: () => void;
  theme: Theme;
}

const defaultTheme = (mode: PaletteMode) =>
  createTheme({
    palette: {
      mode,
      primary: {
        main: mode === 'dark' ? '#6366f1' : '#3f51b5',
        light: mode === 'dark' ? '#818cf8' : '#757de8',
        dark: mode === 'dark' ? '#4f46e5' : '#002984',
        contrastText: '#ffffff',
      },
      secondary: {
        main: mode === 'dark' ? '#10b981' : '#00bcd4',
        light: mode === 'dark' ? '#34d399' : '#62efff',
        dark: mode === 'dark' ? '#059669' : '#008ba3',
        contrastText: '#ffffff',
      },
      background: {
        default: mode === 'dark' ? '#0f172a' : '#f5f5f5',
        paper: mode === 'dark' ? '#1e1e2d' : '#ffffff',
      },
      text: {
        primary: mode === 'dark' ? '#f3f4f6' : 'rgba(0, 0, 0, 0.87)',
        secondary: mode === 'dark' ? '#9ca3af' : 'rgba(0, 0, 0, 0.6)',
      },
      divider: mode === 'dark' ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.12)',
      error: {
        main: mode === 'dark' ? '#ef4444' : '#f44336',
        light: mode === 'dark' ? '#f87171' : '#e57373',
        dark: mode === 'dark' ? '#dc2626' : '#d32f2f',
      },
      warning: {
        main: mode === 'dark' ? '#f59e0b' : '#ff9800',
        light: mode === 'dark' ? '#fbbf24' : '#ffb74d',
        dark: mode === 'dark' ? '#d97706' : '#f57c00',
      },
      info: {
        main: mode === 'dark' ? '#3b82f6' : '#2196f3',
        light: mode === 'dark' ? '#60a5fa' : '#64b5f6',
        dark: mode === 'dark' ? '#2563eb' : '#1976d2',
      },
      success: {
        main: mode === 'dark' ? '#10b981' : '#4caf50',
        light: mode === 'dark' ? '#34d399' : '#81c784',
        dark: mode === 'dark' ? '#059669' : '#388e3c',
      },
    },
    typography: {
      fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
      h1: {
        fontSize: '2.5rem',
        fontWeight: 600,
        lineHeight: 1.2,
      },
      h2: {
        fontSize: '2rem',
        fontWeight: 500,
        lineHeight: 1.3,
      },
      h3: {
        fontSize: '1.75rem',
        fontWeight: 500,
        lineHeight: 1.3,
      },
      h4: {
        fontSize: '1.5rem',
        fontWeight: 500,
        lineHeight: 1.4,
      },
      h5: {
        fontSize: '1.25rem',
        fontWeight: 500,
        lineHeight: 1.4,
      },
      h6: {
        fontSize: '1rem',
        fontWeight: 500,
        lineHeight: 1.5,
      },
      body1: {
        lineHeight: 1.5,
      },
      body2: {
        lineHeight: 1.5,
      },
    },
    components: {
      MuiButton: {
        styleOverrides: {
          root: {
            borderRadius: 8,
            textTransform: 'none',
            fontWeight: 500,
            boxShadow: mode === 'dark' ? '0 4px 12px rgba(0,0,0,0.3)' : '0 4px 12px rgba(0,0,0,0.05)',
          },
          contained: {
            '&:hover': {
              boxShadow: mode === 'dark' ? '0 6px 16px rgba(0,0,0,0.4)' : '0 6px 16px rgba(0,0,0,0.1)',
            },
          },
        },
      },
      MuiCard: {
        styleOverrides: {
          root: {
            borderRadius: 12,
            boxShadow: mode === 'dark' 
              ? '0 4px 12px rgba(0,0,0,0.3)' 
              : '0 4px 12px rgba(0,0,0,0.05)',
          },
        },
      },
      MuiPaper: {
        styleOverrides: {
          root: {
            borderRadius: 8,
          },
        },
      },
      MuiChip: {
        styleOverrides: {
          root: {
            borderRadius: 6,
            fontWeight: 500,
          },
        },
      },
      MuiListItemButton: {
        styleOverrides: {
          root: {
            borderRadius: 8,
            margin: '4px 8px',
            transition: 'all 0.2s ease',
          },
        },
      },
      MuiTableCell: {
        styleOverrides: {
          root: {
            borderBottom: mode === 'dark' 
              ? '1px solid rgba(255, 255, 255, 0.1)' 
              : '1px solid rgba(0, 0, 0, 0.1)',
          },
          head: {
            fontWeight: 600,
          },
        },
      },
      MuiCssBaseline: {
        styleOverrides: {
          body: {
            scrollbarWidth: 'thin',
            scrollbarColor: mode === 'dark' ? '#606060 #383838' : '#c1c1c1 #f1f1f1',
          },
        },
      },
      MuiIconButton: {
        styleOverrides: {
          root: {
            transition: 'all 0.2s ease',
          },
        },
      },
      MuiDrawer: {
        styleOverrides: {
          paper: {
            backgroundImage: mode === 'dark' 
              ? 'linear-gradient(rgba(26, 32, 53, 0.8), rgba(26, 32, 53, 0.8))'
              : 'none',
          },
        },
      },
    },
    shape: {
      borderRadius: 8,
    },
    shadows: [
      'none',
      '0px 2px 4px rgba(0,0,0,0.05)',
      '0px 4px 6px rgba(0,0,0,0.07)',
      '0px 6px 8px rgba(0,0,0,0.08)',
      '0px 8px 10px rgba(0,0,0,0.09)',
      '0px 10px 12px rgba(0,0,0,0.10)',
      '0px 12px 14px rgba(0,0,0,0.11)',
      '0px 14px 16px rgba(0,0,0,0.12)',
      '0px 16px 18px rgba(0,0,0,0.13)',
      '0px 18px 20px rgba(0,0,0,0.14)',
      '0px 20px 22px rgba(0,0,0,0.15)',
      '0px 22px 24px rgba(0,0,0,0.16)',
      '0px 24px 26px rgba(0,0,0,0.17)',
      '0px 26px 28px rgba(0,0,0,0.18)',
      '0px 28px 30px rgba(0,0,0,0.19)',
      '0px 30px 32px rgba(0,0,0,0.20)',
      '0px 32px 34px rgba(0,0,0,0.21)',
      '0px 34px 36px rgba(0,0,0,0.22)',
      '0px 36px 38px rgba(0,0,0,0.23)',
      '0px 38px 40px rgba(0,0,0,0.24)',
      '0px 40px 42px rgba(0,0,0,0.25)',
      '0px 42px 44px rgba(0,0,0,0.26)',
      '0px 44px 46px rgba(0,0,0,0.27)',
      '0px 46px 48px rgba(0,0,0,0.28)',
      '0px 48px 50px rgba(0,0,0,0.29)',
    ],
  });

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export const ThemeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [mode, setMode] = useState<PaletteMode>('dark');
  const [theme, setTheme] = useState<Theme>(defaultTheme(mode));

  useEffect(() => {
    // Load theme preference from localStorage
    const savedMode = localStorage.getItem('themeMode') as PaletteMode | null;
    if (savedMode) {
      setMode(savedMode);
      setTheme(defaultTheme(savedMode));
    } else {
      // Default to dark mode if no preference is saved
      localStorage.setItem('themeMode', 'dark');
    }
  }, []);

  const toggleTheme = () => {
    const newMode = mode === 'light' ? 'dark' : 'light';
    setMode(newMode);
    setTheme(defaultTheme(newMode));
    localStorage.setItem('themeMode', newMode);
  };

  return (
    <ThemeContext.Provider value={{ mode, toggleTheme, theme }}>
      {children}
    </ThemeContext.Provider>
  );
};

export const useTheme = (): ThemeContextType => {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}; 