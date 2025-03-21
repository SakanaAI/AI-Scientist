import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  IconButton,
  Tooltip,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  Pagination,
  CircularProgress,
  Alert,
  Stack
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material';

import PageHeader from '../../components/common/PageHeader';
import { ExperimentCard, ExperimentFilter } from '../../components/experiments';
import experimentService, { Experiment, ExperimentFilter as ExperimentFilterType } from '../../services/experimentService';

const ExperimentsList: React.FC = () => {
  const navigate = useNavigate();
  
  // State for experiments data
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [totalCount, setTotalCount] = useState(0);
  const [totalPages, setTotalPages] = useState(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tags, setTags] = useState<string[]>([]);
  
  // State for pagination and filtering
  const [currentPage, setCurrentPage] = useState(1);
  const [limit] = useState(6);
  const [filters, setFilters] = useState<ExperimentFilterType>({
    status: [],
    dateFrom: null,
    dateTo: null,
    tags: [],
    searchQuery: '',
    page: 1,
    limit: 6
  });
  
  // State for confirmation dialog
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [experimentToDelete, setExperimentToDelete] = useState<string | null>(null);
  
  // Fetch experiments when filters or pagination changes
  useEffect(() => {
    fetchExperiments();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentPage, filters.status, filters.tags, filters.dateFrom, filters.dateTo, filters.searchQuery]);
  
  // Fetch all tags for filter component
  useEffect(() => {
    const fetchTags = async () => {
      try {
        const response = await experimentService.getAllTags();
        setTags(response.data);
      } catch (err) {
        console.error('Error fetching tags:', err);
      }
    };
    
    fetchTags();
  }, []);
  
  const fetchExperiments = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await experimentService.getExperiments({
        ...filters,
        page: currentPage,
        limit
      });
      
      setExperiments(response.data.experiments);
      setTotalCount(response.data.totalCount);
      setTotalPages(response.data.totalPages);
      setCurrentPage(response.data.currentPage);
    } catch (err) {
      console.error('Error fetching experiments:', err);
      setError('Failed to load experiments. Please try again.');
    } finally {
      setLoading(false);
    }
  };
  
  const handlePageChange = (event: React.ChangeEvent<unknown>, page: number) => {
    setCurrentPage(page);
  };
  
  const handleFilterChange = (newFilters: {
    status: string[];
    dateFrom: Date | null;
    dateTo: Date | null;
    tags: string[];
    searchQuery: string;
  }) => {
    setFilters(prev => ({
      ...prev,
      ...newFilters
    }));
    setCurrentPage(1);
  };
  
  const handleRefresh = () => {
    fetchExperiments();
  };
  
  const handleStartExperiment = async (id: string) => {
    try {
      await experimentService.startExperiment(id);
      fetchExperiments();
    } catch (err) {
      console.error('Error starting experiment:', err);
      setError('Failed to start experiment. Please try again.');
    }
  };
  
  const handlePauseExperiment = async (id: string) => {
    try {
      await experimentService.pauseExperiment(id);
      fetchExperiments();
    } catch (err) {
      console.error('Error pausing experiment:', err);
      setError('Failed to pause experiment. Please try again.');
    }
  };
  
  const handleStopExperiment = async (id: string) => {
    try {
      await experimentService.stopExperiment(id);
      fetchExperiments();
    } catch (err) {
      console.error('Error stopping experiment:', err);
      setError('Failed to stop experiment. Please try again.');
    }
  };
  
  const handleDeleteExperiment = async () => {
    if (!experimentToDelete) return;
    
    try {
      await experimentService.deleteExperiment(experimentToDelete);
      setDeleteDialogOpen(false);
      setExperimentToDelete(null);
      fetchExperiments();
    } catch (err) {
      console.error('Error deleting experiment:', err);
      setError('Failed to delete experiment. Please try again.');
    }
  };
  
  const openDeleteDialog = (id: string) => {
    setExperimentToDelete(id);
    setDeleteDialogOpen(true);
  };
  
  // Stats for summary cards
  const completedCount = experiments.filter(exp => exp.status === 'completed').length;
  const runningCount = experiments.filter(exp => exp.status === 'running').length;
  const queuedCount = experiments.filter(exp => exp.status === 'queued').length;
  
  return (
    <Box>
      <PageHeader
        title="Experiments"
        subtitle="Manage and monitor your scientific experiments"
        action={{
          label: "New Experiment",
          icon: <AddIcon />,
          onClick: () => navigate('/experiments/new')
        }}
      />
      
      {/* Stats Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Total Experiments
              </Typography>
              <Typography variant="h4" component="div">
                {totalCount}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Completed
              </Typography>
              <Typography variant="h4" component="div" color="success.main">
                {completedCount}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Running
              </Typography>
              <Typography variant="h4" component="div" color="primary.main">
                {runningCount}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Queued
              </Typography>
              <Typography variant="h4" component="div" color="info.main">
                {queuedCount}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
      
      {/* Filters and Controls */}
      <Box mb={3} display="flex" justifyContent="space-between" alignItems="flex-start">
        <Box flexGrow={1}>
          <ExperimentFilter 
            onFilterChange={handleFilterChange}
            availableTags={tags}
          />
        </Box>
        <Tooltip title="Refresh">
          <IconButton onClick={handleRefresh} sx={{ mt: 1 }}>
            <RefreshIcon />
          </IconButton>
        </Tooltip>
      </Box>
      
      {/* Error Alert */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}
      
      {/* Loading Indicator */}
      {loading ? (
        <Box display="flex" justifyContent="center" my={5}>
          <CircularProgress />
        </Box>
      ) : experiments.length === 0 ? (
        <Box textAlign="center" my={5}>
          <Typography variant="h6" color="text.secondary" gutterBottom>
            No experiments found
          </Typography>
          <Typography variant="body2" color="text.secondary" paragraph>
            Try adjusting your filters or create a new experiment.
          </Typography>
          <Button 
            variant="contained" 
            startIcon={<AddIcon />}
            onClick={() => navigate('/experiments/new')}
          >
            Create Experiment
          </Button>
        </Box>
      ) : (
        <>
          {/* Experiment Cards */}
          <Grid container spacing={3}>
            {experiments.map(experiment => (
              <Grid item xs={12} sm={6} md={4} key={experiment.id}>
                <ExperimentCard
                  experiment={experiment}
                  onStart={handleStartExperiment}
                  onPause={handlePauseExperiment}
                  onStop={handleStopExperiment}
                />
              </Grid>
            ))}
          </Grid>
          
          {/* Pagination */}
          {totalPages > 1 && (
            <Box display="flex" justifyContent="center" mt={4}>
              <Pagination 
                count={totalPages} 
                page={currentPage} 
                onChange={handlePageChange} 
                color="primary" 
                showFirstButton 
                showLastButton
              />
            </Box>
          )}
        </>
      )}
      
      {/* Delete Confirmation Dialog */}
      <Dialog
        open={deleteDialogOpen}
        onClose={() => setDeleteDialogOpen(false)}
      >
        <DialogTitle>Delete Experiment</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to delete this experiment? This action cannot be undone and all associated data will be permanently removed.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleDeleteExperiment} color="error" variant="contained">
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ExperimentsList; 