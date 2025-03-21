import React from 'react';
import { Box, Typography, Button, Breadcrumbs, Link, SxProps, Theme, Paper, alpha } from '@mui/material';
import { Link as RouterLink } from 'react-router-dom';
import { NavigateNext as NavigateNextIcon } from '@mui/icons-material';
import { useTheme } from '../../context/ThemeContext';

interface BreadcrumbItem {
  label: string;
  href?: string;
}

interface PageHeaderProps {
  title: string;
  subtitle?: string;
  breadcrumbs?: BreadcrumbItem[];
  action?: {
    label: string;
    icon?: React.ReactNode;
    onClick: () => void;
  };
  sx?: SxProps<Theme>;
}

const PageHeader: React.FC<PageHeaderProps> = ({
  title,
  subtitle,
  breadcrumbs,
  action,
  sx = {}
}) => {
  const { mode, theme } = useTheme();
  
  return (
    <Paper 
      elevation={0} 
      sx={{ 
        p: { xs: 2, sm: 3 }, 
        mb: 3, 
        background: mode === 'dark' 
          ? alpha(theme.palette.background.paper, 0.6) 
          : alpha(theme.palette.background.paper, 0.8),
        borderRadius: 2,
        backdropFilter: 'blur(8px)',
        ...sx 
      }}
    >
      {breadcrumbs && breadcrumbs.length > 0 && (
        <Breadcrumbs 
          separator={<NavigateNextIcon fontSize="small" sx={{ opacity: 0.5 }} />} 
          aria-label="breadcrumb"
          sx={{ mb: 2 }}
        >
          {breadcrumbs.map((item, index) => {
            const isLast = index === breadcrumbs.length - 1;
            
            return isLast || !item.href ? (
              <Typography color="text.primary" key={index} sx={{ opacity: 0.8, fontSize: '0.875rem' }}>
                {item.label}
              </Typography>
            ) : (
              <Link 
                underline="hover" 
                color="inherit" 
                component={RouterLink} 
                to={item.href} 
                key={index}
                sx={{ opacity: 0.7, '&:hover': { opacity: 1 }, fontSize: '0.875rem' }}
              >
                {item.label}
              </Link>
            );
          })}
        </Breadcrumbs>
      )}
      
      <Box sx={{ 
        display: 'flex', 
        flexDirection: { xs: 'column', sm: 'row' },
        justifyContent: 'space-between', 
        alignItems: { xs: 'flex-start', sm: 'center' },
        gap: { xs: 2, sm: 0 }
      }}>
        <Box>
          <Typography 
            variant="h4" 
            component="h1" 
            fontWeight="600" 
            gutterBottom={!!subtitle} 
            sx={{ 
              fontSize: { xs: '1.75rem', sm: '2rem' },
              lineHeight: 1.2
            }}
          >
            {title}
          </Typography>
          
          {subtitle && (
            <Typography 
              variant="body1" 
              color="text.secondary"
              sx={{ mt: 0.5, maxWidth: '650px' }}
            >
              {subtitle}
            </Typography>
          )}
        </Box>
        
        {action && (
          <Button 
            variant="contained" 
            startIcon={action.icon}
            onClick={action.onClick}
            sx={{ 
              boxShadow: 2,
              px: { xs: 2, sm: 3 },
              py: 1,
              whiteSpace: 'nowrap',
              alignSelf: { xs: 'flex-start', sm: 'center' }
            }}
          >
            {action.label}
          </Button>
        )}
      </Box>
    </Paper>
  );
};

export default PageHeader; 