import React, { useState, useRef, useEffect } from 'react';
import { 
  Box, 
  Paper, 
  Typography, 
  Toolbar, 
  Chip, 
  IconButton, 
  Tooltip, 
  TextField,
  InputAdornment,
  ToggleButtonGroup,
  ToggleButton,
  Button,
  Badge,
  useTheme,
  Stack,
  Divider,
  CircularProgress
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Search as SearchIcon,
  Clear as ClearIcon,
  PlayArrow as PlayArrowIcon,
  Pause as PauseIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
  Info as InfoIcon,
  Code as CodeIcon,
  Download as DownloadIcon
} from '@mui/icons-material';
import { LogEntry } from '../../hooks/useExperimentLogs';

interface ExperimentLogViewerProps {
  logs: LogEntry[];
  loading: boolean;
  isPolling: boolean;
  onRefresh: () => void;
  onClear: () => void;
  onStartPolling: () => void;
  onStopPolling: () => void;
}

const ExperimentLogViewer: React.FC<ExperimentLogViewerProps> = ({
  logs,
  loading,
  isPolling,
  onRefresh,
  onClear,
  onStartPolling,
  onStopPolling
}) => {
  const theme = useTheme();
  const logsEndRef = useRef<HTMLDivElement>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [levelFilter, setLevelFilter] = useState<string[]>([]);
  const [autoScroll, setAutoScroll] = useState(true);
  
  // Filtered logs
  const filteredLogs = logs.filter(log => {
    const matchesSearch = searchQuery === '' || 
      log.message.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (log.source && log.source.toLowerCase().includes(searchQuery.toLowerCase()));
      
    const matchesLevel = levelFilter.length === 0 || levelFilter.includes(log.level);
    
    return matchesSearch && matchesLevel;
  });
  
  // Scroll to bottom when logs change if auto-scroll is enabled
  useEffect(() => {
    if (autoScroll && logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [filteredLogs, autoScroll]);
  
  // Level counts for badges
  const errorCount = logs.filter(log => log.level === 'error').length;
  const warningCount = logs.filter(log => log.level === 'warning').length;
  const infoCount = logs.filter(log => log.level === 'info').length;
  const debugCount = logs.filter(log => log.level === 'debug').length;
  
  const handleLevelFilterChange = (
    event: React.MouseEvent<HTMLElement>,
    newLevels: string[]
  ) => {
    setLevelFilter(newLevels);
  };
  
  // Function to format the timestamp
  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString(undefined, { 
      hour: '2-digit', 
      minute: '2-digit', 
      second: '2-digit',
      hour12: false
    });
  };
  
  // Function to get the color for different log levels
  const getLevelColor = (level: string) => {
    switch (level) {
      case 'error':
        return theme.palette.error.main;
      case 'warning':
        return theme.palette.warning.main;
      case 'info':
        return theme.palette.info.main;
      case 'debug':
        return theme.palette.text.secondary;
      default:
        return theme.palette.text.primary;
    }
  };
  
  // Function to get icon for different log levels
  const getLevelIcon = (level: string) => {
    switch (level) {
      case 'error':
        return <ErrorIcon fontSize="small" sx={{ color: 'error.main' }} />;
      case 'warning':
        return <WarningIcon fontSize="small" sx={{ color: 'warning.main' }} />;
      case 'info':
        return <InfoIcon fontSize="small" sx={{ color: 'info.main' }} />;
      case 'debug':
        return <CodeIcon fontSize="small" sx={{ color: 'text.secondary' }} />;
      default:
        return null;
    }
  };
  
  // Download logs as JSON
  const downloadLogs = () => {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(logs, null, 2));
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", `experiment-logs-${new Date().toISOString()}.json`);
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
  };
  
  return (
    <Paper sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Toolbar with filters and controls */}
      <Toolbar
        variant="dense"
        sx={{
          gap: 1,
          borderBottom: 1,
          borderColor: 'divider',
          display: 'flex',
          flexWrap: 'wrap',
          minHeight: 'auto',
          p: 1
        }}
      >
        <Typography variant="subtitle2" sx={{ mr: 2 }}>
          Logs {filteredLogs.length > 0 && `(${filteredLogs.length})`}
        </Typography>
        
        <Box sx={{ flexGrow: 1, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
          <TextField
            placeholder="Search logs..."
            size="small"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon fontSize="small" />
                </InputAdornment>
              ),
              endAdornment: searchQuery && (
                <InputAdornment position="end">
                  <IconButton
                    size="small"
                    onClick={() => setSearchQuery('')}
                    edge="end"
                  >
                    <ClearIcon fontSize="small" />
                  </IconButton>
                </InputAdornment>
              ),
            }}
            sx={{ minWidth: 200 }}
          />
          
          <ToggleButtonGroup
            size="small"
            value={levelFilter}
            onChange={handleLevelFilterChange}
            aria-label="log level filter"
          >
            <ToggleButton value="error" aria-label="error logs">
              <Badge badgeContent={errorCount} color="error" max={99}>
                <ErrorIcon fontSize="small" />
              </Badge>
            </ToggleButton>
            <ToggleButton value="warning" aria-label="warning logs">
              <Badge badgeContent={warningCount} color="warning" max={99}>
                <WarningIcon fontSize="small" />
              </Badge>
            </ToggleButton>
            <ToggleButton value="info" aria-label="info logs">
              <Badge badgeContent={infoCount} color="info" max={99}>
                <InfoIcon fontSize="small" />
              </Badge>
            </ToggleButton>
            <ToggleButton value="debug" aria-label="debug logs">
              <Badge badgeContent={debugCount} color="default" max={99}>
                <CodeIcon fontSize="small" />
              </Badge>
            </ToggleButton>
          </ToggleButtonGroup>
        </Box>
        
        <Stack direction="row" spacing={1}>
          <Tooltip title={autoScroll ? "Auto-scroll is on" : "Auto-scroll is off"}>
            <Button
              size="small"
              variant={autoScroll ? "contained" : "outlined"}
              color="primary"
              onClick={() => setAutoScroll(!autoScroll)}
            >
              Auto-scroll
            </Button>
          </Tooltip>
          
          <Tooltip title="Download logs">
            <IconButton size="small" onClick={downloadLogs}>
              <DownloadIcon fontSize="small" />
            </IconButton>
          </Tooltip>
          
          <Tooltip title="Clear logs">
            <IconButton size="small" onClick={onClear}>
              <ClearIcon fontSize="small" />
            </IconButton>
          </Tooltip>
          
          <Tooltip title={isPolling ? "Pause live updates" : "Resume live updates"}>
            <IconButton 
              size="small" 
              onClick={isPolling ? onStopPolling : onStartPolling}
              color={isPolling ? "primary" : "default"}
            >
              {isPolling ? <PauseIcon fontSize="small" /> : <PlayArrowIcon fontSize="small" />}
            </IconButton>
          </Tooltip>
          
          <Tooltip title="Refresh logs">
            <IconButton 
              size="small" 
              onClick={onRefresh}
              disabled={loading}
            >
              {loading ? <CircularProgress size={20} /> : <RefreshIcon fontSize="small" />}
            </IconButton>
          </Tooltip>
        </Stack>
      </Toolbar>
      
      {/* Logs content */}
      <Box 
        sx={{ 
          flexGrow: 1, 
          overflow: 'auto', 
          p: 1,
          fontFamily: 'monospace',
          fontSize: '0.875rem',
          bgcolor: theme.palette.mode === 'dark' ? 'background.paper' : 'grey.50'
        }}
      >
        {filteredLogs.length === 0 ? (
          <Box display="flex" justifyContent="center" alignItems="center" height="100%">
            <Typography color="text.secondary">
              {loading ? 'Loading logs...' : 'No logs to display'}
            </Typography>
          </Box>
        ) : (
          filteredLogs.map((log, index) => (
            <Box 
              key={index} 
              sx={{ 
                py: 0.5, 
                borderBottom: index < filteredLogs.length - 1 ? `1px dashed ${theme.palette.divider}` : 'none',
                '&:hover': {
                  bgcolor: theme.palette.action.hover
                }
              }}
            >
              <Box display="flex" alignItems="flex-start">
                <Box 
                  sx={{ 
                    minWidth: 80, 
                    color: 'text.secondary',
                    pr: 1
                  }}
                >
                  {formatTimestamp(log.timestamp)}
                </Box>
                
                <Box display="flex" alignItems="center" mr={1}>
                  {getLevelIcon(log.level)}
                </Box>
                
                {log.source && (
                  <Chip 
                    label={log.source} 
                    size="small" 
                    variant="outlined"
                    sx={{ mr: 1, height: 20, fontSize: '0.75rem' }}
                  />
                )}
                
                <Box 
                  sx={{ 
                    flexGrow: 1,
                    color: getLevelColor(log.level),
                    wordBreak: 'break-word'
                  }}
                >
                  {log.message}
                </Box>
              </Box>
              
              {log.data && (
                <Box 
                  sx={{ 
                    ml: 10, 
                    mt: 0.5, 
                    p: 1, 
                    bgcolor: theme.palette.mode === 'dark' ? 'background.default' : 'grey.100',
                    borderRadius: 1,
                    whiteSpace: 'pre-wrap',
                    fontSize: '0.8rem'
                  }}
                >
                  {typeof log.data === 'string' 
                    ? log.data 
                    : JSON.stringify(log.data, null, 2)
                  }
                </Box>
              )}
            </Box>
          ))
        )}
        <div ref={logsEndRef} />
      </Box>
    </Paper>
  );
};

export default ExperimentLogViewer; 