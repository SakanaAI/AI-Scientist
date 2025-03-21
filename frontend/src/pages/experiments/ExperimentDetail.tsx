import React from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Paper,
  Typography,
  Grid,
  Chip,
  Divider,
  Button,
  Card,
  CardContent,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Tab,
  Tabs,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow
} from '@mui/material';
import {
  Science,
  CheckCircle,
  Pending,
  Info,
  Description,
  BarChart,
  Article,
  ArrowBack,
  Settings,
  Dataset,
  Code,
  Lightbulb
} from '@mui/icons-material';
import PageHeader from '../../components/common/PageHeader';

// Mock experiment data
const experiment = {
  id: '1',
  name: 'Transformer Self-Attention Patterns',
  description: 'This experiment investigates the patterns of self-attention in transformer models and how they correlate with linguistic features.',
  template: 'nanoGPT',
  status: 'completed',
  date: '2023-11-02',
  creator: 'AI-Scientist',
  ideas: 5,
  parameters: {
    'learning_rate': '0.0001',
    'batch_size': '64',
    'epochs': '10',
    'model_size': 'medium',
    'activation': 'gelu'
  },
  metrics: {
    'accuracy': '92.5%',
    'loss': '0.034',
    'training_time': '3h 24m',
    'convergence_epoch': '8'
  },
  ideas_list: [
    { id: '1', title: 'Investigating attention heads specialization in language tasks' },
    { id: '2', title: 'Correlation between attention patterns and syntactic structure' },
    { id: '3', title: 'Effects of different attention mechanisms on model performance' },
    { id: '4', title: 'Visualizing attention patterns for interpretability' },
    { id: '5', title: 'Transfer learning between different attention architectures' }
  ],
  results: [
    { id: '1', name: 'Training Loss Curve', type: 'graph', path: '/results/loss.png' },
    { id: '2', name: 'Attention Visualization', type: 'heatmap', path: '/results/attention.png' },
    { id: '3', name: 'Performance Metrics', type: 'table', path: '/results/metrics.csv' }
  ],
  scripts: [
    { id: '1', name: 'train.py', description: 'Main training script', path: '/scripts/train.py' },
    { id: '2', name: 'model.py', description: 'Model architecture definition', path: '/scripts/model.py' },
    { id: '3', name: 'evaluate.py', description: 'Evaluation script', path: '/scripts/evaluate.py' }
  ]
};

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`experiment-tabpanel-${index}`}
      aria-labelledby={`experiment-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  );
}

const ExperimentDetail: React.FC = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [tabValue, setTabValue] = React.useState(0);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const getStatusChip = (status: string) => {
    switch (status) {
      case 'completed':
        return <Chip icon={<CheckCircle fontSize="small" />} label="Completed" color="success" size="small" />;
      case 'in_progress':
        return <Chip icon={<Pending fontSize="small" />} label="In Progress" color="primary" size="small" />;
      case 'planned':
        return <Chip icon={<Pending fontSize="small" />} label="Planned" color="default" size="small" />;
      default:
        return <Chip label={status} color="default" size="small" />;
    }
  };

  // Breadcrumbs for the page header
  const breadcrumbs = [
    { label: 'Experiments', href: '/experiments' },
    { label: experiment.name }
  ];

  return (
    <Box>
      <PageHeader
        title={experiment.name}
        subtitle={`Template: ${experiment.template} • Created: ${experiment.date}`}
        breadcrumbs={breadcrumbs}
        action={{
          label: "Back to Experiments",
          icon: <ArrowBack />,
          onClick: () => navigate('/experiments')
        }}
      />

      {/* Status and Quick Info */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item>
            <Science fontSize="large" color="primary" />
          </Grid>
          <Grid item xs>
            <Typography variant="h6">{experiment.name}</Typography>
            <Typography variant="body2" color="text.secondary">
              {experiment.description}
            </Typography>
          </Grid>
          <Grid item>
            {getStatusChip(experiment.status)}
          </Grid>
          <Grid item>
            <Button 
              variant="outlined" 
              startIcon={<Description />}
              onClick={() => navigate('/papers/new')}
            >
              Generate Paper
            </Button>
          </Grid>
        </Grid>
      </Paper>

      {/* Tabs Navigation */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs 
          value={tabValue} 
          onChange={handleTabChange} 
          aria-label="experiment tabs"
          variant="scrollable"
          scrollButtons="auto"
        >
          <Tab icon={<Info />} label="Overview" />
          <Tab icon={<Lightbulb />} label="Research Ideas" />
          <Tab icon={<BarChart />} label="Results" />
          <Tab icon={<Code />} label="Code" />
          <Tab icon={<Dataset />} label="Data" />
          <Tab icon={<Settings />} label="Settings" />
        </Tabs>
      </Box>

      {/* Overview Tab */}
      <TabPanel value={tabValue} index={0}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Experiment Parameters
                </Typography>
                <Divider sx={{ mb: 2 }} />
                <Grid container spacing={2}>
                  {Object.entries(experiment.parameters).map(([key, value]) => (
                    <Grid item xs={6} key={key}>
                      <Typography variant="subtitle2" color="text.secondary">
                        {key}
                      </Typography>
                      <Typography variant="body1">
                        {value}
                      </Typography>
                    </Grid>
                  ))}
                </Grid>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Performance Metrics
                </Typography>
                <Divider sx={{ mb: 2 }} />
                <Grid container spacing={2}>
                  {Object.entries(experiment.metrics).map(([key, value]) => (
                    <Grid item xs={6} key={key}>
                      <Typography variant="subtitle2" color="text.secondary">
                        {key}
                      </Typography>
                      <Typography variant="body1">
                        {value}
                      </Typography>
                    </Grid>
                  ))}
                </Grid>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Timeline
                </Typography>
                <Divider sx={{ mb: 2 }} />
                <List>
                  <ListItem>
                    <ListItemIcon>
                      <CheckCircle color="success" />
                    </ListItemIcon>
                    <ListItemText 
                      primary="Experiment Started" 
                      secondary={`${experiment.date} - Ideas generated: ${experiment.ideas}`} 
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <CheckCircle color="success" />
                    </ListItemIcon>
                    <ListItemText 
                      primary="Data Processing Completed" 
                      secondary="2023-11-03 - Processed 2.4GB of data" 
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <CheckCircle color="success" />
                    </ListItemIcon>
                    <ListItemText 
                      primary="Training Completed" 
                      secondary="2023-11-05 - Model converged after 8 epochs" 
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <CheckCircle color="success" />
                    </ListItemIcon>
                    <ListItemText 
                      primary="Evaluation Completed" 
                      secondary="2023-11-06 - Final accuracy: 92.5%" 
                    />
                  </ListItem>
                </List>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Research Ideas Tab */}
      <TabPanel value={tabValue} index={1}>
        <Typography variant="h6" gutterBottom>
          Research Ideas Generated
        </Typography>
        <Typography variant="body2" color="text.secondary" paragraph>
          These research ideas were automatically generated by the AI-Scientist based on the experiment template and domain.
        </Typography>
        
        <Grid container spacing={2}>
          {experiment.ideas_list.map((idea) => (
            <Grid item xs={12} md={6} key={idea.id}>
              <Card>
                <CardContent>
                  <Typography variant="subtitle1" gutterBottom>
                    <Lightbulb color="primary" sx={{ mr: 1, verticalAlign: 'middle' }} />
                    {idea.title}
                  </Typography>
                  <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
                    <Button size="small">Explore</Button>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </TabPanel>

      {/* Results Tab */}
      <TabPanel value={tabValue} index={2}>
        <Typography variant="h6" gutterBottom>
          Experiment Results
        </Typography>
        <Typography variant="body2" color="text.secondary" paragraph>
          These are the results and outputs generated from running the experiment.
        </Typography>
        
        <TableContainer component={Paper} sx={{ mb: 3 }}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Name</TableCell>
                <TableCell>Type</TableCell>
                <TableCell>Path</TableCell>
                <TableCell align="right">Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {experiment.results.map((result) => (
                <TableRow key={result.id}>
                  <TableCell>{result.name}</TableCell>
                  <TableCell>{result.type}</TableCell>
                  <TableCell>{result.path}</TableCell>
                  <TableCell align="right">
                    <Button size="small">View</Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
        
        <Box sx={{ textAlign: 'center' }}>
          <Button 
            variant="contained" 
            startIcon={<BarChart />}
            onClick={() => navigate('/results')}
          >
            View Detailed Analysis
          </Button>
        </Box>
      </TabPanel>

      {/* Code Tab */}
      <TabPanel value={tabValue} index={3}>
        <Typography variant="h6" gutterBottom>
          Experiment Code
        </Typography>
        <Typography variant="body2" color="text.secondary" paragraph>
          These are the scripts and code files used to run this experiment.
        </Typography>
        
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Script Name</TableCell>
                <TableCell>Description</TableCell>
                <TableCell>Path</TableCell>
                <TableCell align="right">Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {experiment.scripts.map((script) => (
                <TableRow key={script.id}>
                  <TableCell>{script.name}</TableCell>
                  <TableCell>{script.description}</TableCell>
                  <TableCell>{script.path}</TableCell>
                  <TableCell align="right">
                    <Button size="small">View</Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </TabPanel>

      {/* Data Tab */}
      <TabPanel value={tabValue} index={4}>
        <Typography variant="h6" gutterBottom>
          Experiment Data
        </Typography>
        <Typography variant="body2" color="text.secondary" paragraph>
          This is the data used and generated during the experiment.
        </Typography>
        
        <Box sx={{ mb: 3 }}>
          <Card>
            <CardContent>
              <Typography variant="subtitle1" gutterBottom>
                Data Summary
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6} md={3}>
                  <Typography variant="subtitle2" color="text.secondary">
                    Total Dataset Size
                  </Typography>
                  <Typography variant="body1">
                    2.4 GB
                  </Typography>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Typography variant="subtitle2" color="text.secondary">
                    Number of Samples
                  </Typography>
                  <Typography variant="body1">
                    125,000
                  </Typography>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Typography variant="subtitle2" color="text.secondary">
                    Training Set
                  </Typography>
                  <Typography variant="body1">
                    100,000 samples
                  </Typography>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Typography variant="subtitle2" color="text.secondary">
                    Validation Set
                  </Typography>
                  <Typography variant="body1">
                    25,000 samples
                  </Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Box>
        
        <Button variant="outlined" startIcon={<Dataset />}>
          Download Full Dataset
        </Button>
      </TabPanel>

      {/* Settings Tab */}
      <TabPanel value={tabValue} index={5}>
        <Typography variant="h6" gutterBottom>
          Experiment Settings
        </Typography>
        <Typography variant="body2" color="text.secondary" paragraph>
          These are the configuration settings used for this experiment.
        </Typography>
        
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="subtitle1" gutterBottom>
              Environment Configuration
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <Typography variant="subtitle2" color="text.secondary">
                  Compute Resources
                </Typography>
                <Typography variant="body1">
                  4 GPUs (NVIDIA A100)
                </Typography>
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography variant="subtitle2" color="text.secondary">
                  Framework Version
                </Typography>
                <Typography variant="body1">
                  PyTorch 2.0.1
                </Typography>
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography variant="subtitle2" color="text.secondary">
                  Python Version
                </Typography>
                <Typography variant="body1">
                  Python 3.9.7
                </Typography>
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography variant="subtitle2" color="text.secondary">
                  CUDA Version
                </Typography>
                <Typography variant="body1">
                  CUDA 11.7
                </Typography>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
        
        <Button variant="outlined" startIcon={<Settings />}>
          Edit Configuration
        </Button>
      </TabPanel>
    </Box>
  );
};

export default ExperimentDetail; 