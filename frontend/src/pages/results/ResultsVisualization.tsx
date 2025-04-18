import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  SelectChangeEvent,
  Divider,
  Button,
  Tabs,
  Tab,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Chip,
  ToggleButtonGroup,
  ToggleButton,
  CircularProgress
} from '@mui/material';
import {
  BarChart as BarChartIcon,
  PieChart as PieChartIcon,
  ShowChart as LineChartIcon,
  BubbleChart as ScatterPlotIcon,
  TableChart,
  Lightbulb,
  Science,
  Compare,
  Download,
  FilterList,
  ArrowForward,
  CloudDownload,
  Timeline
} from '@mui/icons-material';
import PageHeader from '../../components/common/PageHeader';

// For a real implementation, we would use a charting library like Chart.js or Recharts
// This is just a placeholder for demonstration purposes
const ChartPlaceholder: React.FC<{ type: string; height?: number; color?: string }> = ({ 
  type, 
  height = 300, 
  color = 'primary.main' 
}) => {
  return (
    <Box 
      sx={{ 
        height, 
        bgcolor: 'grey.100', 
        display: 'flex', 
        flexDirection: 'column',
        alignItems: 'center', 
        justifyContent: 'center', 
        borderRadius: 1,
        border: '1px dashed',
        borderColor: 'grey.400'
      }}
    >
      {type === 'bar' && <BarChartIcon sx={{ fontSize: 60, color }} />}
      {type === 'line' && <LineChartIcon sx={{ fontSize: 60, color }} />}
      {type === 'pie' && <PieChartIcon sx={{ fontSize: 60, color }} />}
      {type === 'scatter' && <ScatterPlotIcon sx={{ fontSize: 60, color }} />}
      <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
        {type.charAt(0).toUpperCase() + type.slice(1)} Chart Visualization
      </Typography>
    </Box>
  );
};

// Mock experiment data for visualization
const experiments = [
  { id: '1', name: 'Transformer Self-Attention Patterns', template: 'nanoGPT' },
  { id: '2', name: 'Efficient 2D Diffusion Model Training', template: '2d_diffusion' },
  { id: '3', name: 'Effects of Learning Rate on Grokking', template: 'grokking' },
  { id: '4', name: 'Analysis of RLHF Training', template: 'custom' }
];

// Mock metrics for the selected experiment
const metrics = [
  { id: '1', name: 'Training Loss', type: 'line', description: 'Loss value during training' },
  { id: '2', name: 'Validation Accuracy', type: 'line', description: 'Accuracy on validation set' },
  { id: '3', name: 'Parameter Distribution', type: 'pie', description: 'Distribution of model parameters by layer' },
  { id: '4', name: 'Learning Rate vs. Performance', type: 'scatter', description: 'Relationship between learning rate and model performance' },
  { id: '5', name: 'Attention Head Analysis', type: 'bar', description: 'Contribution of different attention heads to model performance' }
];

// Mock comparison metrics
const comparisonMetrics = [
  { id: '1', name: 'Training Time', unit: 'hours', values: [3.2, 5.1, 2.8, 4.5] },
  { id: '2', name: 'Max Accuracy', unit: '%', values: [92.3, 88.7, 95.1, 91.0] },
  { id: '3', name: 'Model Size', unit: 'MB', values: [250, 180, 310, 220] },
  { id: '4', name: 'GPU Hours', unit: '', values: [12.5, 18.2, 10.3, 15.7] }
];

// Mock insight data
const insights = [
  { id: '1', text: 'Training convergence occurs faster with cosine learning rate schedule', importance: 'high' },
  { id: '2', text: 'Attention heads 3, 7, and 12 show strong specialization for syntactic features', importance: 'medium' },
  { id: '3', text: 'Model performance plateaus after 8 training epochs', importance: 'medium' },
  { id: '4', text: 'Larger batch sizes correlate with more stable training dynamics', importance: 'high' },
  { id: '5', text: 'Layer normalization significantly improves model convergence', importance: 'medium' }
];

type ChartType = 'line' | 'bar' | 'pie' | 'scatter';

const ResultsVisualization: React.FC = () => {
  const navigate = useNavigate();
  
  // State
  const [selectedExperiment, setSelectedExperiment] = useState('1');
  const [selectedMetric, setSelectedMetric] = useState('1');
  const [tabValue, setTabValue] = useState(0);
  const [chartType, setChartType] = useState<ChartType>('line');
  const [isLoading, setIsLoading] = useState(false);
  
  const handleExperimentChange = (event: SelectChangeEvent) => {
    setIsLoading(true);
    setSelectedExperiment(event.target.value);
    // Simulate loading data
    setTimeout(() => {
      setIsLoading(false);
    }, 800);
  };
  
  const handleMetricChange = (event: SelectChangeEvent) => {
    setSelectedMetric(event.target.value);
    // Set chart type based on the selected metric
    const metric = metrics.find(m => m.id === event.target.value);
    if (metric) {
      setChartType(metric.type as ChartType);
    }
  };
  
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };
  
  const handleChartTypeChange = (
    event: React.MouseEvent<HTMLElement>,
    newChartType: ChartType,
  ) => {
    if (newChartType !== null) {
      setChartType(newChartType);
    }
  };
  
  // Get the selected experiment name
  const getExperimentName = () => {
    const experiment = experiments.find(exp => exp.id === selectedExperiment);
    return experiment ? experiment.name : '';
  };
  
  // Get the selected metric name
  const getMetricName = () => {
    const metric = metrics.find(m => m.id === selectedMetric);
    return metric ? metric.name : '';
  };
  
  // Get the selected metric description
  const getMetricDescription = () => {
    const metric = metrics.find(m => m.id === selectedMetric);
    return metric ? metric.description : '';
  };
  
  // Breadcrumbs for the page header
  const breadcrumbs = [
    { label: 'Results Visualization' }
  ];
  
  return (
    <Box>
      <PageHeader
        title="Results Visualization"
        subtitle="Analyze and visualize experiment results"
        breadcrumbs={breadcrumbs}
      />
      
      {/* Main Content */}
      <Grid container spacing={3}>
        {/* Left Sidebar: Filters and Controls */}
        <Grid item xs={12} md={4} lg={3}>
          <Paper sx={{ p: 2, mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Visualization Controls
            </Typography>
            <Divider sx={{ mb: 2 }} />
            
            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel id="experiment-select-label">Experiment</InputLabel>
              <Select
                labelId="experiment-select-label"
                id="experiment-select"
                value={selectedExperiment}
                label="Experiment"
                onChange={handleExperimentChange}
              >
                {experiments.map((experiment) => (
                  <MenuItem key={experiment.id} value={experiment.id}>
                    {experiment.name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            
            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel id="metric-select-label">Metric</InputLabel>
              <Select
                labelId="metric-select-label"
                id="metric-select"
                value={selectedMetric}
                label="Metric"
                onChange={handleMetricChange}
              >
                {metrics.map((metric) => (
                  <MenuItem key={metric.id} value={metric.id}>
                    {metric.name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            
            <Typography variant="subtitle2" gutterBottom>
              Chart Type
            </Typography>
            <ToggleButtonGroup
              value={chartType}
              exclusive
              onChange={handleChartTypeChange}
              aria-label="chart type"
              size="small"
              sx={{ mb: 2, display: 'flex', flexWrap: 'wrap' }}
            >
              <ToggleButton value="line" aria-label="line chart">
                <LineChartIcon fontSize="small" />
              </ToggleButton>
              <ToggleButton value="bar" aria-label="bar chart">
                <BarChartIcon fontSize="small" />
              </ToggleButton>
              <ToggleButton value="pie" aria-label="pie chart">
                <PieChartIcon fontSize="small" />
              </ToggleButton>
              <ToggleButton value="scatter" aria-label="scatter plot">
                <ScatterPlotIcon fontSize="small" />
              </ToggleButton>
            </ToggleButtonGroup>
            
            <Divider sx={{ my: 2 }} />
            
            <Button 
              variant="outlined" 
              fullWidth 
              startIcon={<CloudDownload />}
              onClick={() => {}}
              sx={{ mb: 1 }}
            >
              Export Chart
            </Button>
            
            <Button 
              variant="outlined" 
              fullWidth 
              startIcon={<Download />}
              onClick={() => {}}
            >
              Download Raw Data
            </Button>
          </Paper>
          
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Key Insights
            </Typography>
            <Divider sx={{ mb: 2 }} />
            
            <List dense>
              {insights.map((insight) => (
                <ListItem key={insight.id} sx={{ px: 0 }}>
                  <ListItemIcon>
                    <Lightbulb color="primary" />
                  </ListItemIcon>
                  <ListItemText 
                    primary={insight.text} 
                    secondary={`Importance: ${insight.importance}`} 
                  />
                </ListItem>
              ))}
            </List>
            
            <Divider sx={{ my: 2 }} />
            
            <Button 
              fullWidth 
              variant="contained" 
              endIcon={<ArrowForward />}
              onClick={() => navigate('/experiments')}
            >
              New Experiment
            </Button>
          </Paper>
        </Grid>
        
        {/* Main Content Area */}
        <Grid item xs={12} md={8} lg={9}>
          <Paper sx={{ mb: 3 }}>
            <Tabs
              value={tabValue}
              onChange={handleTabChange}
              aria-label="result visualization tabs"
              sx={{ borderBottom: 1, borderColor: 'divider', px: 2 }}
            >
              <Tab icon={<BarChartIcon />} label="Visualization" />
              <Tab icon={<Compare />} label="Comparison" />
              <Tab icon={<TableChart />} label="Data Table" />
              <Tab icon={<Timeline />} label="Timeline" />
            </Tabs>
            
            {/* Visualization Tab */}
            {tabValue === 0 && (
              <Box sx={{ p: 3 }}>
                {isLoading ? (
                  <Box sx={{ display: 'flex', justifyContent: 'center', py: 10 }}>
                    <CircularProgress />
                  </Box>
                ) : (
                  <>
                    <Box sx={{ mb: 3 }}>
                      <Typography variant="h5" gutterBottom>
                        {getMetricName()}
                      </Typography>
                      <Typography variant="subtitle1" color="text.secondary" gutterBottom>
                        {getExperimentName()}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {getMetricDescription()}
                      </Typography>
                    </Box>
                    
                    <ChartPlaceholder type={chartType} height={400} />
                    
                    <Box sx={{ mt: 2, display: 'flex', justifyContent: 'flex-end' }}>
                      <Button startIcon={<FilterList />} sx={{ mr: 1 }}>
                        Apply Filters
                      </Button>
                      <Button startIcon={<CloudDownload />}>
                        Download
                      </Button>
                    </Box>
                  </>
                )}
              </Box>
            )}
            
            {/* Comparison Tab */}
            {tabValue === 1 && (
              <Box sx={{ p: 3 }}>
                <Typography variant="h5" gutterBottom>
                  Experiment Comparison
                </Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                  Compare key metrics across different experiments
                </Typography>
                
                <Grid container spacing={3}>
                  {comparisonMetrics.map((metric) => (
                    <Grid item xs={12} sm={6} key={metric.id}>
                      <Card>
                        <CardContent>
                          <Typography variant="h6" gutterBottom>
                            {metric.name}
                          </Typography>
                          
                          <ChartPlaceholder type="bar" height={250} />
                          
                          <Grid container spacing={1} sx={{ mt: 2 }}>
                            {experiments.map((exp, index) => (
                              <Grid item xs={6} key={exp.id}>
                                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                  <Typography variant="body2" color="text.secondary">
                                    {exp.name.substring(0, 15)}...
                                  </Typography>
                                  <Typography variant="body2" fontWeight="medium">
                                    {metric.values[index]}{metric.unit}
                                  </Typography>
                                </Box>
                              </Grid>
                            ))}
                          </Grid>
                        </CardContent>
                      </Card>
                    </Grid>
                  ))}
                </Grid>
              </Box>
            )}
            
            {/* Data Table Tab */}
            {tabValue === 2 && (
              <Box sx={{ p: 3 }}>
                <Typography variant="h5" gutterBottom>
                  Raw Data Table
                </Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                  View and download the raw data for detailed analysis
                </Typography>
                
                <Paper sx={{ p: 2, bgcolor: 'grey.50' }}>
                  <Typography variant="body2" color="text.secondary" align="center">
                    Data table visualization would be displayed here.
                  </Typography>
                  <Typography variant="body2" color="text.secondary" align="center">
                    This would typically include a data grid component for browsing and filtering the raw data.
                  </Typography>
                </Paper>
                
                <Box sx={{ mt: 2, display: 'flex', justifyContent: 'flex-end' }}>
                  <Button startIcon={<Download />}>
                    Export to CSV
                  </Button>
                </Box>
              </Box>
            )}
            
            {/* Timeline Tab */}
            {tabValue === 3 && (
              <Box sx={{ p: 3 }}>
                <Typography variant="h5" gutterBottom>
                  Experiment Timeline
                </Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                  View the progress and key events of the experiment over time
                </Typography>
                
                <ChartPlaceholder type="line" height={200} color="secondary.main" />
                
                <Box sx={{ mt: 3 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    Key Events
                  </Typography>
                  <List>
                    <ListItem>
                      <ListItemIcon>
                        <Science color="primary" />
                      </ListItemIcon>
                      <ListItemText 
                        primary="Experiment Started" 
                        secondary="2023-10-25 14:32:15" 
                      />
                      <Chip label="Day 1" size="small" />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        <Science color="primary" />
                      </ListItemIcon>
                      <ListItemText 
                        primary="Data Processing Completed" 
                        secondary="2023-10-25 18:45:22" 
                      />
                      <Chip label="Day 1" size="small" />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        <Science color="primary" />
                      </ListItemIcon>
                      <ListItemText 
                        primary="Training Started" 
                        secondary="2023-10-26 09:10:05" 
                      />
                      <Chip label="Day 2" size="small" />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        <Science color="success" />
                      </ListItemIcon>
                      <ListItemText 
                        primary="Training Completed" 
                        secondary="2023-10-28 22:17:33" 
                      />
                      <Chip label="Day 4" size="small" />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        <Science color="success" />
                      </ListItemIcon>
                      <ListItemText 
                        primary="Evaluation Completed" 
                        secondary="2023-10-29 15:30:45" 
                      />
                      <Chip label="Day 5" size="small" />
                    </ListItem>
                  </List>
                </Box>
              </Box>
            )}
          </Paper>
          
          {/* Recent Results */}
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Related Visualizations
            </Typography>
            <Divider sx={{ mb: 2 }} />
            
            <Grid container spacing={2}>
              {metrics.slice(0, 3).map((metric) => (
                <Grid item xs={12} md={4} key={metric.id}>
                  <Card sx={{ height: '100%' }}>
                    <CardContent>
                      <Typography variant="subtitle1" gutterBottom>
                        {metric.name}
                      </Typography>
                      <ChartPlaceholder type={metric.type as ChartType} height={150} />
                      <Button 
                        size="small" 
                        sx={{ mt: 1 }}
                        onClick={() => setSelectedMetric(metric.id)}
                      >
                        View Details
                      </Button>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default ResultsVisualization; 