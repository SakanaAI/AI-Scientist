import React from 'react';
import { Card, CardContent, Typography, Box, Chip, LinearProgress, IconButton, Tooltip } from '@mui/material';
import { PlayArrow, Pause, Stop, Visibility } from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

interface ExperimentCardProps {
  experiment: {
    id: string;
    title: string;
    description: string;
    status: 'running' | 'completed' | 'failed' | 'paused' | 'queued';
    progress: number;
    createdAt: string;
    tags: string[];
  };
  onStart?: (id: string) => void;
  onPause?: (id: string) => void;
  onStop?: (id: string) => void;
}

const getStatusColor = (status: string) => {
  switch (status) {
    case 'running': return 'primary';
    case 'completed': return 'success';
    case 'failed': return 'error';
    case 'paused': return 'warning';
    case 'queued': return 'info';
    default: return 'default';
  }
};

const ExperimentCard: React.FC<ExperimentCardProps> = ({ experiment, onStart, onPause, onStop }) => {
  const navigate = useNavigate();
  
  const viewExperiment = () => {
    navigate(`/experiments/${experiment.id}`);
  };

  return (
    <Card 
      sx={{ 
        mb: 2, 
        transition: 'transform 0.2s', 
        '&:hover': { transform: 'translateY(-3px)', boxShadow: 3 }
      }}
    >
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
          <Typography variant="h6" noWrap sx={{ maxWidth: '70%' }}>
            {experiment.title}
          </Typography>
          <Chip 
            label={experiment.status.toUpperCase()} 
            color={getStatusColor(experiment.status) as any} 
            size="small" 
          />
        </Box>
        
        <Typography variant="body2" color="text.secondary" mb={2} sx={{
          display: '-webkit-box',
          WebkitLineClamp: 2,
          WebkitBoxOrient: 'vertical',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          height: '40px'
        }}>
          {experiment.description}
        </Typography>
        
        <Box mb={1}>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={0.5}>
            <Typography variant="body2">Progress</Typography>
            <Typography variant="body2">{Math.round(experiment.progress)}%</Typography>
          </Box>
          <LinearProgress 
            variant="determinate" 
            value={experiment.progress} 
            sx={{ height: 8, borderRadius: 4 }}
          />
        </Box>
        
        <Box display="flex" flexWrap="wrap" gap={0.5} mb={2}>
          {experiment.tags.map((tag, index) => (
            <Chip 
              key={index} 
              label={tag} 
              size="small" 
              sx={{ fontSize: '0.7rem' }}
            />
          ))}
        </Box>
        
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Typography variant="caption" color="text.secondary">
            {new Date(experiment.createdAt).toLocaleDateString()}
          </Typography>
          
          <Box>
            {experiment.status !== 'completed' && experiment.status !== 'failed' && (
              <>
                {experiment.status !== 'running' && onStart && (
                  <Tooltip title="Start">
                    <IconButton size="small" onClick={() => onStart(experiment.id)}>
                      <PlayArrow fontSize="small" />
                    </IconButton>
                  </Tooltip>
                )}
                
                {experiment.status === 'running' && onPause && (
                  <Tooltip title="Pause">
                    <IconButton size="small" onClick={() => onPause(experiment.id)}>
                      <Pause fontSize="small" />
                    </IconButton>
                  </Tooltip>
                )}
                
                {onStop && (
                  <Tooltip title="Stop">
                    <IconButton size="small" onClick={() => onStop(experiment.id)}>
                      <Stop fontSize="small" />
                    </IconButton>
                  </Tooltip>
                )}
              </>
            )}
            
            <Tooltip title="View Details">
              <IconButton size="small" onClick={viewExperiment}>
                <Visibility fontSize="small" />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};

export default ExperimentCard; 