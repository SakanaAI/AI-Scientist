import React, { useState } from 'react';
import { Outlet } from 'react-router-dom';
import {
  Box,
  Drawer,
  AppBar,
  Toolbar,
  Typography,
  Divider,
  IconButton,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  useMediaQuery,
  useTheme as useMuiTheme,
  Avatar,
  Menu,
  MenuItem,
  Tooltip,
  Badge,
  alpha
} from '@mui/material';
import {
  Menu as MenuIcon,
  Dashboard as DashboardIcon,
  Science as ScienceIcon,
  Description as DescriptionIcon,
  BarChart as BarChartIcon,
  Settings as SettingsIcon,
  LightMode as LightModeIcon,
  DarkMode as DarkModeIcon,
  Notifications as NotificationsIcon,
  AccountCircle,
  ChevronLeft as ChevronLeftIcon
} from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import { useTheme } from '../../context/ThemeContext';

const drawerWidth = 260;

interface NavItem {
  text: string;
  icon: React.ReactNode;
  path: string;
}

const MainLayout: React.FC = () => {
  const { logout, user } = useAuth();
  const { toggleTheme, mode } = useTheme();
  const muiTheme = useMuiTheme();
  const navigate = useNavigate();
  const location = useLocation();
  const isMobile = useMediaQuery(muiTheme.breakpoints.down('md'));
  
  const [open, setOpen] = useState(!isMobile);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [notificationsAnchorEl, setNotificationsAnchorEl] = useState<null | HTMLElement>(null);
  
  const handleDrawerToggle = () => {
    setOpen(!open);
  };
  
  const handleProfileMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };
  
  const handleProfileMenuClose = () => {
    setAnchorEl(null);
  };
  
  const handleNotificationsMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setNotificationsAnchorEl(event.currentTarget);
  };
  
  const handleNotificationsMenuClose = () => {
    setNotificationsAnchorEl(null);
  };
  
  const handleLogout = () => {
    handleProfileMenuClose();
    logout();
    navigate('/auth/login');
  };
  
  const navItems: NavItem[] = [
    { text: 'Dashboard', icon: <DashboardIcon />, path: '/' },
    { text: 'Experiments', icon: <ScienceIcon />, path: '/experiments' },
    { text: 'Papers', icon: <DescriptionIcon />, path: '/papers' },
    { text: 'Results', icon: <BarChartIcon />, path: '/results' }
  ];
  
  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      <AppBar 
        position="fixed" 
        elevation={0}
        sx={{ 
          zIndex: (theme) => theme.zIndex.drawer + 1,
          width: { md: open ? `calc(100% - ${drawerWidth}px)` : '100%' },
          ml: { md: open ? `${drawerWidth}px` : 0 },
          transition: muiTheme.transitions.create(['width', 'margin'], {
            easing: muiTheme.transitions.easing.sharp,
            duration: muiTheme.transitions.duration.leavingScreen,
          }),
          backdropFilter: 'blur(8px)',
          backgroundColor: mode === 'dark' 
            ? alpha(muiTheme.palette.background.paper, 0.9)
            : alpha(muiTheme.palette.background.paper, 0.9),
          borderBottom: `1px solid ${muiTheme.palette.divider}`
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2 }}
          >
            {open ? <ChevronLeftIcon /> : <MenuIcon />}
          </IconButton>
          <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1, fontWeight: 600 }}>
            AI-Scientist
          </Typography>
          
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Tooltip title={`Switch to ${mode === 'light' ? 'dark' : 'light'} mode`}>
              <IconButton 
                color="inherit" 
                onClick={toggleTheme}
                sx={{ mx: 1 }}
              >
                {mode === 'light' ? <DarkModeIcon /> : <LightModeIcon />}
              </IconButton>
            </Tooltip>
            
            <Tooltip title="Notifications">
              <IconButton 
                color="inherit"
                onClick={handleNotificationsMenuOpen}
                sx={{ mx: 1 }}
              >
                <Badge badgeContent={3} color="error">
                  <NotificationsIcon />
                </Badge>
              </IconButton>
            </Tooltip>
            
            <Tooltip title="Account">
              <IconButton
                edge="end"
                aria-label="account of current user"
                aria-controls="menu-appbar"
                aria-haspopup="true"
                onClick={handleProfileMenuOpen}
                color="inherit"
                sx={{ ml: 1 }}
              >
                {user?.username ? (
                  <Avatar sx={{ width: 32, height: 32, bgcolor: 'primary.main' }}>
                    {user.username.charAt(0).toUpperCase()}
                  </Avatar>
                ) : (
                  <AccountCircle />
                )}
              </IconButton>
            </Tooltip>
          </Box>
          
          <Menu
            id="menu-appbar"
            anchorEl={anchorEl}
            anchorOrigin={{
              vertical: 'bottom',
              horizontal: 'right',
            }}
            keepMounted
            transformOrigin={{
              vertical: 'top',
              horizontal: 'right',
            }}
            open={Boolean(anchorEl)}
            onClose={handleProfileMenuClose}
          >
            <MenuItem onClick={() => { handleProfileMenuClose(); navigate('/profile'); }}>
              Profile
            </MenuItem>
            <MenuItem onClick={() => { handleProfileMenuClose(); navigate('/settings'); }}>
              Settings
            </MenuItem>
            <Divider />
            <MenuItem onClick={handleLogout}>Logout</MenuItem>
          </Menu>
          
          <Menu
            id="notifications-menu"
            anchorEl={notificationsAnchorEl}
            anchorOrigin={{
              vertical: 'bottom',
              horizontal: 'right',
            }}
            keepMounted
            transformOrigin={{
              vertical: 'top',
              horizontal: 'right',
            }}
            open={Boolean(notificationsAnchorEl)}
            onClose={handleNotificationsMenuClose}
            PaperProps={{ sx: { width: 320, maxHeight: 500 } }}
          >
            <Box sx={{ p: 2, borderBottom: `1px solid ${muiTheme.palette.divider}` }}>
              <Typography variant="subtitle1" fontWeight={600}>Notifications</Typography>
            </Box>
            <MenuItem onClick={handleNotificationsMenuClose}>
              <ListItemIcon>
                <ScienceIcon color="primary" fontSize="small" />
              </ListItemIcon>
              <ListItemText 
                primary="Experiment Completed" 
                secondary="NanoGPT experiment finished successfully" 
              />
            </MenuItem>
            <MenuItem onClick={handleNotificationsMenuClose}>
              <ListItemIcon>
                <DescriptionIcon color="info" fontSize="small" />
              </ListItemIcon>
              <ListItemText 
                primary="Paper Generated" 
                secondary="A new paper draft has been created" 
              />
            </MenuItem>
            <MenuItem onClick={handleNotificationsMenuClose}>
              <ListItemIcon>
                <NotificationsIcon color="error" fontSize="small" />
              </ListItemIcon>
              <ListItemText 
                primary="System Alert" 
                secondary="Resources usage exceeding 80%" 
              />
            </MenuItem>
            <Divider />
            <Box sx={{ p: 1, textAlign: 'center' }}>
              <Typography 
                variant="body2" 
                color="primary" 
                sx={{ cursor: 'pointer', '&:hover': { textDecoration: 'underline' } }}
              >
                View all notifications
              </Typography>
            </Box>
          </Menu>
        </Toolbar>
      </AppBar>
      
      <Drawer
        variant={isMobile ? "temporary" : "permanent"}
        open={isMobile ? open : true}
        onClose={isMobile ? handleDrawerToggle : undefined}
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          [`& .MuiDrawer-paper`]: { 
            width: drawerWidth, 
            boxSizing: 'border-box',
            borderRight: `1px solid ${muiTheme.palette.divider}`
          },
        }}
      >
        <Toolbar sx={{ 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center',
          py: 1.5,
          px: 2
        }}>
          <Typography variant="h5" component="div" sx={{ fontWeight: 700, color: 'primary.main' }}>
            AI-Scientist
          </Typography>
        </Toolbar>
        <Divider />
        <Box sx={{ overflow: 'auto', height: '100%', pt: 1 }}>
          <List>
            {navItems.map((item) => {
              const isActive = location.pathname === item.path || 
                (item.path !== '/' && location.pathname.startsWith(item.path));
                
              return (
                <ListItem key={item.text} disablePadding>
                  <ListItemButton 
                    onClick={() => navigate(item.path)}
                    selected={isActive}
                    sx={{
                      borderRadius: '0 24px 24px 0',
                      mx: 1,
                      my: 0.5,
                      '&.Mui-selected': {
                        backgroundColor: 'primary.main',
                        color: 'primary.contrastText',
                        '& .MuiListItemIcon-root': {
                          color: 'primary.contrastText',
                        },
                        '&:hover': {
                          backgroundColor: 'primary.dark',
                        },
                      },
                    }}
                  >
                    <ListItemIcon sx={{ 
                      color: isActive ? 'primary.contrastText' : 'inherit',
                      minWidth: 40
                    }}>
                      {item.icon}
                    </ListItemIcon>
                    <ListItemText primary={item.text} />
                    {isActive && (
                      <Box 
                        component="span" 
                        sx={{ 
                          width: 4, 
                          height: 32, 
                          backgroundColor: 'primary.contrastText',
                          position: 'absolute',
                          right: 0,
                          borderRadius: '4px 0 0 4px',
                          opacity: 0.8
                        }} 
                      />
                    )}
                  </ListItemButton>
                </ListItem>
              );
            })}
          </List>
          <Divider sx={{ my: 2 }} />
          <List>
            <ListItem disablePadding>
              <ListItemButton
                onClick={() => navigate('/settings')}
                sx={{ borderRadius: '0 24px 24px 0', mx: 1, my: 0.5 }}
              >
                <ListItemIcon sx={{ minWidth: 40 }}>
                  <SettingsIcon />
                </ListItemIcon>
                <ListItemText primary="Settings" />
              </ListItemButton>
            </ListItem>
          </List>
        </Box>
      </Drawer>
      
      <Box component="main" sx={{ 
        flexGrow: 1, 
        p: 3,
        width: { md: open ? `calc(100% - ${drawerWidth}px)` : '100%' },
        ml: { md: open ? `${drawerWidth}px` : 0 },
        transition: muiTheme.transitions.create(['width', 'margin'], {
          easing: muiTheme.transitions.easing.sharp,
          duration: muiTheme.transitions.duration.leavingScreen,
        }),
      }}>
        <Toolbar />
        <Outlet />
      </Box>
    </Box>
  );
};

export default MainLayout; 