import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Grid,
  Typography,
  Card,
  CardContent,
  CardActions,
  Button,
  Divider,
  Paper,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Chip,
  LinearProgress
} from '@mui/material';
import {
  Science,
  Description,
  Pending,
  CheckCircle,
  Warning,
  Lightbulb,
  AddCircle,
  BarChart,
  History
} from '@mui/icons-material';

// Mock data for the dashboard
const recentExperiments = [
  { id: '1', name: 'Transformer Self-Attention Patterns', template: 'nanoGPT', status: 'completed', date: '2023-11-02' },
  { id: '2', name: 'Efficient 2D Diffusion Model Training', template: '2d_diffusion', status: 'in_progress', date: '2023-11-01' },
  { id: '3', name: 'Effects of Learning Rate on Grokking', template: 'grokking', status: 'planned', date: '2023-10-28' }
];

const recentPapers = [
  { id: '1', title: 'Understanding Emergent Abilities in LLMs', status: 'published', date: '2023-10-30' },
  { id: '2', title: 'Efficient Image Generation with Diffusion Models', status: 'draft', date: '2023-10-25' }
];

const recentIdeas = [
  { id: '1', title: 'Investigating Scaling Laws for Multimodal Models', score: 92 },
  { id: '2', title: 'Analyzing Attention Patterns in Vision Transformers', score: 87 },
  { id: '3', title: 'Optimizing Diffusion Model Sampling Efficiency', score: 85 }
];

// Dashboard stats
const stats = {
  totalExperiments: 12,
  completedExperiments: 8,
  inProgressExperiments: 2,
  totalPapers: 5,
  totalIdeas: 24
};

const Dashboard: React.FC = () => {
  const navigate = useNavigate();
  
  const getStatusChip = (status: string) => {
    switch (status) {
      case 'completed':
        return <Chip icon={<CheckCircle fontSize="small" />} label="Completed" color="success" size="small" />;
      case 'in_progress':
        return <Chip icon={<Pending fontSize="small" />} label="In Progress" color="primary" size="small" />;
      case 'planned':
        return <Chip icon={<Pending fontSize="small" />} label="Planned" color="default" size="small" />;
      case 'published':
        return <Chip icon={<CheckCircle fontSize="small" />} label="Published" color="success" size="small" />;
      case 'draft':
        return <Chip icon={<Description fontSize="small" />} label="Draft" color="info" size="small" />;
      default:
        return <Chip icon={<Warning fontSize="small" />} label="Unknown" color="default" size="small" />;
    }
  };
  
  return (
    <Box>
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom fontWeight="500">
          Dashboard
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Overview of research activities and progress
        </Typography>
      </Box>
      
      {/* Stats Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography color="text.secondary" variant="subtitle2">
                  Total Experiments
                </Typography>
                <Science color="primary" />
              </Box>
              <Typography variant="h4" component="div" sx={{ mt: 1, fontWeight: 600 }}>
                {stats.totalExperiments}
              </Typography>
              <Box sx={{ mt: 2, mb: 1 }}>
                <Typography variant="body2" component="span">
                  {stats.completedExperiments} completed
                </Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={(stats.completedExperiments / stats.totalExperiments) * 100} 
                  sx={{ mt: 1 }}
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography color="text.secondary" variant="subtitle2">
                  Total Papers
                </Typography>
                <Description color="primary" />
              </Box>
              <Typography variant="h4" component="div" sx={{ mt: 1, fontWeight: 600 }}>
                {stats.totalPapers}
              </Typography>
              <Box sx={{ mt: 3 }}>
                <Button 
                  size="small" 
                  startIcon={<AddCircle />}
                  onClick={() => navigate('/papers')}
                >
                  View All Papers
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography color="text.secondary" variant="subtitle2">
                  Total Ideas
                </Typography>
                <Lightbulb color="primary" />
              </Box>
              <Typography variant="h4" component="div" sx={{ mt: 1, fontWeight: 600 }}>
                {stats.totalIdeas}
              </Typography>
              <Box sx={{ mt: 3 }}>
                <Button 
                  size="small" 
                  startIcon={<AddCircle />}
                  onClick={() => {}}
                >
                  Generate New Ideas
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography color="text.secondary" variant="subtitle2">
                  Active Runs
                </Typography>
                <Pending color="primary" />
              </Box>
              <Typography variant="h4" component="div" sx={{ mt: 1, fontWeight: 600 }}>
                {stats.inProgressExperiments}
              </Typography>
              <Box sx={{ mt: 3 }}>
                <Button 
                  size="small" 
                  startIcon={<History />}
                  onClick={() => navigate('/experiments')}
                >
                  View Active Runs
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
      
      <Grid container spacing={3}>
        {/* Recent Experiments */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">Recent Experiments</Typography>
              <Button 
                size="small" 
                endIcon={<AddCircle />}
                onClick={() => navigate('/experiments/new')}
              >
                New Experiment
              </Button>
            </Box>
            <Divider sx={{ mb: 2 }} />
            <List>
              {recentExperiments.map((exp) => (
                <ListItem 
                  key={exp.id}
                  secondaryAction={getStatusChip(exp.status)}
                  sx={{ px: 1 }}
                >
                  <ListItemIcon>
                    <Science color="primary" />
                  </ListItemIcon>
                  <ListItemText 
                    primary={exp.name} 
                    secondary={`${exp.template} • ${exp.date}`}
                    primaryTypographyProps={{ fontWeight: 500 }}
                  />
                </ListItem>
              ))}
            </List>
            <Divider sx={{ mt: 2, mb: 2 }} />
            <Box sx={{ textAlign: 'center' }}>
              <Button 
                variant="outlined" 
                size="small"
                onClick={() => navigate('/experiments')}
              >
                View All Experiments
              </Button>
            </Box>
          </Paper>
        </Grid>
        
        {/* Recent Papers */}
        <Grid item xs={12} md={6}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Paper sx={{ p: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6">Recent Papers</Typography>
                  <Button 
                    size="small" 
                    endIcon={<AddCircle />}
                    onClick={() => navigate('/papers')}
                  >
                    View All
                  </Button>
                </Box>
                <Divider sx={{ mb: 2 }} />
                <List>
                  {recentPapers.map((paper) => (
                    <ListItem 
                      key={paper.id}
                      secondaryAction={getStatusChip(paper.status)}
                      sx={{ px: 1 }}
                    >
                      <ListItemIcon>
                        <Description color="primary" />
                      </ListItemIcon>
                      <ListItemText 
                        primary={paper.title} 
                        secondary={`Last updated: ${paper.date}`}
                        primaryTypographyProps={{ fontWeight: 500 }}
                      />
                    </ListItem>
                  ))}
                </List>
              </Paper>
            </Grid>
            
            {/* Top Ideas */}
            <Grid item xs={12}>
              <Paper sx={{ p: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6">Top Research Ideas</Typography>
                  <Button 
                    size="small" 
                    endIcon={<BarChart />}
                    onClick={() => navigate('/results')}
                  >
                    Analysis
                  </Button>
                </Box>
                <Divider sx={{ mb: 2 }} />
                <List>
                  {recentIdeas.map((idea) => (
                    <ListItem 
                      key={idea.id}
                      sx={{ px: 1 }}
                      secondaryAction={
                        <Chip 
                          label={`Score: ${idea.score}%`} 
                          color={idea.score > 90 ? 'success' : 'primary'} 
                          size="small" 
                        />
                      }
                    >
                      <ListItemIcon>
                        <Lightbulb color="primary" />
                      </ListItemIcon>
                      <ListItemText 
                        primary={idea.title}
                        primaryTypographyProps={{ fontWeight: 500 }}
                      />
                    </ListItem>
                  ))}
                </List>
              </Paper>
            </Grid>
          </Grid>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard; 