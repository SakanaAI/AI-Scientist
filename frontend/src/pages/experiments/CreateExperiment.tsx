import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Divider,
  Card,
  CardContent,
  CardMedia,
  FormControlLabel,
  Switch,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  FormHelperText,
  Chip,
  Alert,
  AlertTitle,
  CircularProgress,
  SelectChangeEvent
} from '@mui/material';
import {
  Science,
  ArrowBack,
  ArrowForward,
  Check,
  Settings,
  Memory,
  Lightbulb
} from '@mui/icons-material';
import PageHeader from '../../components/common/PageHeader';

interface DefaultParams {
  learningRate: string;
  batchSize: string;
  epochs: string;
  modelSize: string;
  activation?: string;
  noise_steps?: string;
  weight_decay?: string;
  [key: string]: string | undefined;
}

interface TemplateType {
  id: string;
  name: string;
  description: string;
  image: string;
  domains: string[];
  defaultParams: DefaultParams;
}

// Template information
const templates: TemplateType[] = [
  {
    id: 'nanoGPT',
    name: 'NanoGPT',
    description: 'Experiments with transformer language models and their properties',
    image: 'https://via.placeholder.com/300x140?text=NanoGPT',
    domains: ['NLP', 'Language Models', 'Transformers'],
    defaultParams: {
      learningRate: '0.0001',
      batchSize: '64',
      epochs: '10',
      modelSize: 'medium',
      activation: 'gelu'
    }
  },
  {
    id: '2d_diffusion',
    name: '2D Diffusion',
    description: 'Diffusion models for image generation and manipulation',
    image: 'https://via.placeholder.com/300x140?text=2D+Diffusion',
    domains: ['Computer Vision', 'Generative Models', 'Diffusion'],
    defaultParams: {
      learningRate: '0.00002',
      batchSize: '32',
      epochs: '20',
      modelSize: 'medium',
      noise_steps: '1000'
    }
  },
  {
    id: 'grokking',
    name: 'Grokking',
    description: 'Investigate the grokking phenomenon in neural networks',
    image: 'https://via.placeholder.com/300x140?text=Grokking',
    domains: ['Deep Learning', 'Optimization', 'Generalization'],
    defaultParams: {
      learningRate: '0.0001',
      batchSize: '128',
      epochs: '50',
      modelSize: 'small',
      weight_decay: '0.01'
    }
  },
  {
    id: 'custom',
    name: 'Custom Experiment',
    description: 'Define your own experimental setup from scratch',
    image: 'https://via.placeholder.com/300x140?text=Custom+Experiment',
    domains: ['Custom'],
    defaultParams: {
      learningRate: '0.001',
      batchSize: '32',
      epochs: '10',
      modelSize: 'medium'
    }
  }
];

// Available GPU resources
const gpuOptions = [
  { value: '1', label: '1 GPU' },
  { value: '2', label: '2 GPUs' },
  { value: '4', label: '4 GPUs' },
  { value: '8', label: '8 GPUs' }
];

// Available LLM models
const llmOptions = [
  { value: 'claude-3-5-sonnet-20241022', label: 'Claude 3.5 Sonnet' },
  { value: 'gpt-4o-2024-05-13', label: 'GPT-4o' },
  { value: 'claude-3-opus-20240229', label: 'Claude 3 Opus' },
  { value: 'gpt-4-1106-preview', label: 'GPT-4 Turbo' }
];

// Steps for the experiment creation process
const steps = [
  {
    label: 'Select Template',
    description: 'Choose an experiment template to use as a starting point'
  },
  {
    label: 'Configure Parameters',
    description: 'Set up the experiment parameters and configuration'
  },
  {
    label: 'Resources & LLM',
    description: 'Configure compute resources and LLM for idea generation'
  },
  {
    label: 'Review & Create',
    description: 'Review your experiment settings and create the experiment'
  }
];

const CreateExperiment: React.FC = () => {
  const navigate = useNavigate();
  
  // State for stepper
  const [activeStep, setActiveStep] = useState(0);
  
  // State for form values
  const [selectedTemplate, setSelectedTemplate] = useState<string>('');
  const [experimentName, setExperimentName] = useState('');
  const [experimentDescription, setExperimentDescription] = useState('');
  const [numIdeas, setNumIdeas] = useState('5');
  const [parameters, setParameters] = useState<Record<string, string>>({});
  const [selectedGpu, setSelectedGpu] = useState('2');
  const [selectedModel, setSelectedModel] = useState('claude-3-5-sonnet-20241022');
  const [advancedMode, setAdvancedMode] = useState(false);
  const [loading, setLoading] = useState(false);
  
  // Handler for template selection
  const handleTemplateChange = (templateId: string) => {
    setSelectedTemplate(templateId);
    const template = templates.find(t => t.id === templateId);
    
    if (template) {
      setExperimentName(`${template.name} Experiment`);
      setExperimentDescription(template.description);
      
      // Convert defaultParams to Record<string, string> by filtering out undefined values
      const params: Record<string, string> = {};
      Object.entries(template.defaultParams).forEach(([key, value]) => {
        if (value !== undefined) {
          params[key] = value;
        }
      });
      setParameters(params);
    }
  };
  
  // Handler for next step
  const handleNext = () => {
    setActiveStep((prevActiveStep) => prevActiveStep + 1);
  };
  
  // Handler for back step
  const handleBack = () => {
    setActiveStep((prevActiveStep) => prevActiveStep - 1);
  };
  
  // Handler for parameter change
  const handleParameterChange = (key: string, value: string) => {
    setParameters(prev => ({
      ...prev,
      [key]: value
    }));
  };
  
  // Handler for GPU selection
  const handleGpuChange = (event: SelectChangeEvent) => {
    setSelectedGpu(event.target.value);
  };
  
  // Handler for LLM model selection
  const handleModelChange = (event: SelectChangeEvent) => {
    setSelectedModel(event.target.value);
  };
  
  // Handler for form submission
  const handleSubmit = () => {
    setLoading(true);
    
    // Simulate API call
    setTimeout(() => {
      setLoading(false);
      navigate('/experiments');
    }, 2000);
  };
  
  // Check if current step is valid
  const isStepValid = () => {
    switch (activeStep) {
      case 0:
        return !!selectedTemplate;
      case 1:
        return !!experimentName && !!experimentDescription;
      case 2:
        return !!selectedGpu && !!selectedModel && !!numIdeas;
      default:
        return true;
    }
  };
  
  // Get the selected template object
  const getSelectedTemplate = () => {
    return templates.find(t => t.id === selectedTemplate);
  };
  
  // Breadcrumbs for the page header
  const breadcrumbs = [
    { label: 'Experiments', href: '/experiments' },
    { label: 'New Experiment' }
  ];
  
  return (
    <Box>
      <PageHeader
        title="Create New Experiment"
        subtitle="Set up a new scientific experiment"
        breadcrumbs={breadcrumbs}
        action={{
          label: "Back to Experiments",
          icon: <ArrowBack />,
          onClick: () => navigate('/experiments')
        }}
      />
      
      <Grid container spacing={3}>
        {/* Left side: Stepper */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3, mb: { xs: 3, md: 0 } }}>
            <Stepper activeStep={activeStep} orientation="vertical">
              {steps.map((step, index) => (
                <Step key={step.label}>
                  <StepLabel>
                    <Typography variant="subtitle1">{step.label}</Typography>
                  </StepLabel>
                  <StepContent>
                    <Typography variant="body2" color="text.secondary">
                      {step.description}
                    </Typography>
                    <Box sx={{ mb: 2, mt: 2 }}>
                      <Button
                        variant="contained"
                        onClick={handleNext}
                        disabled={!isStepValid()}
                        sx={{ mt: 1, mr: 1 }}
                        endIcon={activeStep === steps.length - 1 ? <Check /> : <ArrowForward />}
                      >
                        {activeStep === steps.length - 1 ? 'Create' : 'Continue'}
                      </Button>
                      <Button
                        disabled={activeStep === 0}
                        onClick={handleBack}
                        sx={{ mt: 1, mr: 1 }}
                      >
                        Back
                      </Button>
                    </Box>
                  </StepContent>
                </Step>
              ))}
            </Stepper>
          </Paper>
        </Grid>
        
        {/* Right side: Form content */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3 }}>
            {/* Step 1: Select Template */}
            {activeStep === 0 && (
              <Box>
                <Typography variant="h6" gutterBottom>
                  Select Experiment Template
                </Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                  Choose a template to use as a starting point for your experiment.
                </Typography>
                
                <Grid container spacing={2}>
                  {templates.map((template) => (
                    <Grid item xs={12} sm={6} key={template.id}>
                      <Card 
                        sx={{ 
                          cursor: 'pointer',
                          border: selectedTemplate === template.id ? 2 : 0,
                          borderColor: 'primary.main',
                          height: '100%',
                          display: 'flex',
                          flexDirection: 'column'
                        }}
                        onClick={() => handleTemplateChange(template.id)}
                      >
                        <CardMedia
                          component="img"
                          height="140"
                          image={template.image || "https://via.placeholder.com/300x140"}
                          alt={template.name}
                        />
                        <CardContent sx={{ flexGrow: 1 }}>
                          <Typography variant="h6" component="div" gutterBottom>
                            {template.name}
                          </Typography>
                          <Typography variant="body2" color="text.secondary" paragraph>
                            {template.description}
                          </Typography>
                          <Box>
                            {template.domains.map(domain => (
                              <Chip 
                                key={domain} 
                                label={domain} 
                                size="small" 
                                sx={{ mr: 0.5, mb: 0.5 }}
                              />
                            ))}
                          </Box>
                        </CardContent>
                      </Card>
                    </Grid>
                  ))}
                </Grid>
              </Box>
            )}
            
            {/* Step 2: Configure Parameters */}
            {activeStep === 1 && (
              <Box>
                <Typography variant="h6" gutterBottom>
                  Configure Experiment
                </Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                  Set up the basic information and parameters for your experiment.
                </Typography>
                
                <Grid container spacing={3}>
                  <Grid item xs={12}>
                    <TextField
                      fullWidth
                      label="Experiment Name"
                      value={experimentName}
                      onChange={(e) => setExperimentName(e.target.value)}
                      required
                    />
                  </Grid>
                  
                  <Grid item xs={12}>
                    <TextField
                      fullWidth
                      label="Description"
                      value={experimentDescription}
                      onChange={(e) => setExperimentDescription(e.target.value)}
                      multiline
                      rows={3}
                      required
                    />
                  </Grid>
                  
                  <Grid item xs={12}>
                    <Divider sx={{ my: 2 }}>
                      <Chip 
                        label="Experiment Parameters" 
                        icon={<Settings fontSize="small" />} 
                      />
                    </Divider>
                  </Grid>
                  
                  <Grid item xs={12}>
                    <FormControlLabel
                      control={
                        <Switch 
                          checked={advancedMode} 
                          onChange={(e) => setAdvancedMode(e.target.checked)} 
                        />
                      }
                      label="Advanced Mode"
                    />
                  </Grid>
                  
                  {Object.entries(parameters).map(([key, value]) => (
                    <Grid item xs={12} sm={6} md={4} key={key}>
                      <TextField
                        fullWidth
                        label={key.charAt(0).toUpperCase() + key.slice(1).replace(/([A-Z])/g, ' $1')}
                        value={value}
                        onChange={(e) => handleParameterChange(key, e.target.value)}
                        disabled={!advancedMode}
                      />
                    </Grid>
                  ))}
                </Grid>
              </Box>
            )}
            
            {/* Step 3: Resources & LLM */}
            {activeStep === 2 && (
              <Box>
                <Typography variant="h6" gutterBottom>
                  Configure Resources & LLM
                </Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                  Set up the compute resources and LLM for idea generation.
                </Typography>
                
                <Grid container spacing={3}>
                  <Grid item xs={12} md={6}>
                    <Card sx={{ height: '100%' }}>
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          <Memory fontSize="small" sx={{ verticalAlign: 'middle', mr: 1 }} />
                          Compute Resources
                        </Typography>
                        
                        <FormControl fullWidth sx={{ mb: 2 }}>
                          <InputLabel id="gpu-label">GPU Resources</InputLabel>
                          <Select
                            labelId="gpu-label"
                            value={selectedGpu}
                            label="GPU Resources"
                            onChange={handleGpuChange}
                          >
                            {gpuOptions.map(option => (
                              <MenuItem key={option.value} value={option.value}>
                                {option.label}
                              </MenuItem>
                            ))}
                          </Select>
                          <FormHelperText>
                            Select the number of GPUs to allocate for this experiment
                          </FormHelperText>
                        </FormControl>
                        
                        <Alert severity="info" sx={{ mt: 2 }}>
                          <AlertTitle>Resource Usage</AlertTitle>
                          Your current resource allocation: <strong>2/8 GPUs (25%)</strong>
                        </Alert>
                      </CardContent>
                    </Card>
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <Card sx={{ height: '100%' }}>
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          <Lightbulb fontSize="small" sx={{ verticalAlign: 'middle', mr: 1 }} />
                          Idea Generation
                        </Typography>
                        
                        <FormControl fullWidth sx={{ mb: 2 }}>
                          <InputLabel id="llm-label">LLM Model</InputLabel>
                          <Select
                            labelId="llm-label"
                            value={selectedModel}
                            label="LLM Model"
                            onChange={handleModelChange}
                          >
                            {llmOptions.map(option => (
                              <MenuItem key={option.value} value={option.value}>
                                {option.label}
                              </MenuItem>
                            ))}
                          </Select>
                          <FormHelperText>
                            Select the LLM model for generating research ideas
                          </FormHelperText>
                        </FormControl>
                        
                        <TextField
                          fullWidth
                          label="Number of Ideas"
                          type="number"
                          InputProps={{ inputProps: { min: 1, max: 10 } }}
                          value={numIdeas}
                          onChange={(e) => setNumIdeas(e.target.value)}
                          helperText="How many research ideas to generate (1-10)"
                        />
                      </CardContent>
                    </Card>
                  </Grid>
                </Grid>
              </Box>
            )}
            
            {/* Step 4: Review & Create */}
            {activeStep === 3 && (
              <Box>
                <Typography variant="h6" gutterBottom>
                  Review & Create Experiment
                </Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                  Review your experiment configuration and create the experiment.
                </Typography>
                
                <Card sx={{ mb: 3 }}>
                  <CardContent>
                    <Grid container spacing={2}>
                      <Grid item xs={12} sm={6}>
                        <Typography variant="subtitle2" color="text.secondary">
                          Experiment Name
                        </Typography>
                        <Typography variant="body1" gutterBottom>
                          {experimentName}
                        </Typography>
                      </Grid>
                      
                      <Grid item xs={12} sm={6}>
                        <Typography variant="subtitle2" color="text.secondary">
                          Template
                        </Typography>
                        <Typography variant="body1" gutterBottom>
                          {getSelectedTemplate()?.name}
                        </Typography>
                      </Grid>
                      
                      <Grid item xs={12}>
                        <Typography variant="subtitle2" color="text.secondary">
                          Description
                        </Typography>
                        <Typography variant="body1" gutterBottom>
                          {experimentDescription}
                        </Typography>
                      </Grid>
                      
                      <Grid item xs={12}>
                        <Divider sx={{ my: 1 }} />
                      </Grid>
                      
                      <Grid item xs={12} sm={6}>
                        <Typography variant="subtitle2" color="text.secondary">
                          GPU Resources
                        </Typography>
                        <Typography variant="body1" gutterBottom>
                          {gpuOptions.find(opt => opt.value === selectedGpu)?.label}
                        </Typography>
                      </Grid>
                      
                      <Grid item xs={12} sm={6}>
                        <Typography variant="subtitle2" color="text.secondary">
                          LLM Model
                        </Typography>
                        <Typography variant="body1" gutterBottom>
                          {llmOptions.find(opt => opt.value === selectedModel)?.label}
                        </Typography>
                      </Grid>
                      
                      <Grid item xs={12} sm={6}>
                        <Typography variant="subtitle2" color="text.secondary">
                          Number of Ideas
                        </Typography>
                        <Typography variant="body1" gutterBottom>
                          {numIdeas}
                        </Typography>
                      </Grid>
                      
                      <Grid item xs={12}>
                        <Divider sx={{ my: 1 }} />
                      </Grid>
                      
                      <Grid item xs={12}>
                        <Typography variant="subtitle2" color="text.secondary">
                          Parameters
                        </Typography>
                        <Grid container spacing={1} sx={{ mt: 1 }}>
                          {Object.entries(parameters).map(([key, value]) => (
                            <Grid item xs={6} sm={4} key={key}>
                              <Typography variant="caption" color="text.secondary">
                                {key.charAt(0).toUpperCase() + key.slice(1).replace(/([A-Z])/g, ' $1')}
                              </Typography>
                              <Typography variant="body2">
                                {value}
                              </Typography>
                            </Grid>
                          ))}
                        </Grid>
                      </Grid>
                    </Grid>
                  </CardContent>
                </Card>
                
                <Box sx={{ display: 'flex', justifyContent: 'center' }}>
                  <Button
                    variant="contained"
                    size="large"
                    onClick={handleSubmit}
                    startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <Science />}
                    disabled={loading}
                    sx={{ px: 4, py: 1 }}
                  >
                    {loading ? 'Creating Experiment...' : 'Create Experiment'}
                  </Button>
                </Box>
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default CreateExperiment; 