import { useState, useCallback, useEffect } from 'react';
import experimentService, { 
  Experiment, 
  ExperimentFilter
} from '../services/experimentService';

interface UseExperimentsProps {
  initialFilters?: ExperimentFilter;
}

export const useExperiments = (props?: UseExperimentsProps) => {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [experiment, setExperiment] = useState<Experiment | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [totalCount, setTotalCount] = useState(0);
  const [totalPages, setTotalPages] = useState(1);
  const [currentPage, setCurrentPage] = useState(1);
  const [limit] = useState(6);
  const [filters, setFilters] = useState<ExperimentFilter>(props?.initialFilters || {
    status: [],
    dateFrom: null,
    dateTo: null,
    tags: [],
    searchQuery: '',
    page: 1,
    limit: 6
  });
  const [availableTags, setAvailableTags] = useState<string[]>([]);

  const fetchExperiments = useCallback(async (page?: number, newFilters?: Partial<ExperimentFilter>) => {
    setLoading(true);
    setError(null);
    
    try {
      const currentFilters = {
        ...filters,
        ...(newFilters || {}),
        page: page || filters.page || 1,
        limit: filters.limit || limit
      };
      
      const response = await experimentService.getExperiments(currentFilters);
      
      setExperiments(response.data.experiments);
      setTotalCount(response.data.totalCount);
      setTotalPages(response.data.totalPages);
      setCurrentPage(response.data.currentPage);
      
      return response.data;
    } catch (err) {
      console.error('Error fetching experiments:', err);
      setError('Failed to load experiments. Please try again.');
      return null;
    } finally {
      setLoading(false);
    }
  }, [filters, limit]);

  const fetchExperiment = useCallback(async (id: string) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await experimentService.getExperiment(id);
      setExperiment(response.data);
      return response.data;
    } catch (err) {
      console.error('Error fetching experiment:', err);
      setError('Failed to load experiment details. Please try again.');
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const createExperiment = useCallback(async (data: {
    title: string;
    description: string;
    parameters: Record<string, any>;
    tags: string[];
    paperUrl?: string;
  }) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await experimentService.createExperiment(data);
      return response.data;
    } catch (err) {
      console.error('Error creating experiment:', err);
      setError('Failed to create experiment. Please try again.');
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const updateExperiment = useCallback(async (id: string, data: Partial<Experiment>) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await experimentService.updateExperiment(id, data);
      
      if (experiment && experiment.id === id) {
        setExperiment(response.data);
      }
      
      return response.data;
    } catch (err) {
      console.error('Error updating experiment:', err);
      setError('Failed to update experiment. Please try again.');
      return null;
    } finally {
      setLoading(false);
    }
  }, [experiment]);

  const deleteExperiment = useCallback(async (id: string) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await experimentService.deleteExperiment(id);
      
      if (experiment && experiment.id === id) {
        setExperiment(null);
      }
      
      return response.data;
    } catch (err) {
      console.error('Error deleting experiment:', err);
      setError('Failed to delete experiment. Please try again.');
      return null;
    } finally {
      setLoading(false);
    }
  }, [experiment]);

  const startExperiment = useCallback(async (id: string) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await experimentService.startExperiment(id);
      
      if (experiment && experiment.id === id) {
        setExperiment(response.data);
      }
      
      return response.data;
    } catch (err) {
      console.error('Error starting experiment:', err);
      setError('Failed to start experiment. Please try again.');
      return null;
    } finally {
      setLoading(false);
    }
  }, [experiment]);

  const pauseExperiment = useCallback(async (id: string) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await experimentService.pauseExperiment(id);
      
      if (experiment && experiment.id === id) {
        setExperiment(response.data);
      }
      
      return response.data;
    } catch (err) {
      console.error('Error pausing experiment:', err);
      setError('Failed to pause experiment. Please try again.');
      return null;
    } finally {
      setLoading(false);
    }
  }, [experiment]);

  const stopExperiment = useCallback(async (id: string) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await experimentService.stopExperiment(id);
      
      if (experiment && experiment.id === id) {
        setExperiment(response.data);
      }
      
      return response.data;
    } catch (err) {
      console.error('Error stopping experiment:', err);
      setError('Failed to stop experiment. Please try again.');
      return null;
    } finally {
      setLoading(false);
    }
  }, [experiment]);

  const fetchTags = useCallback(async () => {
    try {
      const response = await experimentService.getAllTags();
      setAvailableTags(response.data);
      return response.data;
    } catch (err) {
      console.error('Error fetching tags:', err);
      return [];
    }
  }, []);

  const handleFilterChange = useCallback((newFilters: Partial<ExperimentFilter>) => {
    setFilters(prev => ({
      ...prev,
      ...newFilters,
      page: 1 // Reset to first page when changing filters
    }));
  }, []);

  const handlePageChange = useCallback((page: number) => {
    setCurrentPage(page);
    setFilters(prev => ({
      ...prev,
      page
    }));
  }, []);

  // Fetch tags on initial load
  useEffect(() => {
    fetchTags();
  }, [fetchTags]);

  return {
    // Data
    experiments,
    experiment,
    loading,
    error,
    totalCount,
    totalPages,
    currentPage,
    limit,
    filters,
    availableTags,
    
    // Methods
    fetchExperiments,
    fetchExperiment,
    createExperiment,
    updateExperiment,
    deleteExperiment,
    startExperiment,
    pauseExperiment,
    stopExperiment,
    fetchTags,
    handleFilterChange,
    handlePageChange
  };
};

export default useExperiments; 