import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

// Types
export interface Experiment {
  id: string;
  title: string;
  description: string;
  status: 'running' | 'completed' | 'failed' | 'paused' | 'queued';
  progress: number;
  createdAt: string;
  updatedAt: string;
  userId: string;
  tags: string[];
  parameters: Record<string, any>;
  results?: any;
  logs?: string[];
  steps?: {
    id: string;
    label: string;
    description: string;
    status: 'completed' | 'in-progress' | 'pending' | 'failed';
    estimatedTimeRemaining?: string;
    completedAt?: string;
  }[];
  paperUrl?: string;
  resources?: {
    cpu: string;
    memory: string;
    gpu?: string;
  };
  estimatedTimeRemaining?: string;
  startTime?: string;
  endTime?: string;
}

export interface ExperimentFilter {
  status?: string[];
  dateFrom?: Date | null;
  dateTo?: Date | null;
  tags?: string[];
  searchQuery?: string;
  page?: number;
  limit?: number;
  sortBy?: string;
  sortDirection?: 'asc' | 'desc';
}

export interface CreateExperimentData {
  title: string;
  description: string;
  parameters: Record<string, any>;
  tags: string[];
  paperUrl?: string;
}

// Mock data & helper functions (will be replaced with actual API calls)
let mockExperiments: Experiment[] = Array.from({ length: 20 }).map((_, index) => ({
  id: `exp-${index + 1}`,
  title: `Experiment ${index + 1}`,
  description: `This is a description for experiment ${index + 1}. It contains details about the experiment's purpose and methodology.`,
  status: ['running', 'completed', 'failed', 'paused', 'queued'][Math.floor(Math.random() * 5)] as any,
  progress: Math.random() * 100,
  createdAt: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000).toISOString(),
  updatedAt: new Date(Date.now() - Math.random() * 15 * 24 * 60 * 60 * 1000).toISOString(),
  userId: 'user-1',
  tags: ['ML', 'Neural Networks', 'Research', 'GPT', 'NLP', 'Computer Vision'].sort(() => 0.5 - Math.random()).slice(0, Math.floor(Math.random() * 4) + 1),
  parameters: {
    learningRate: 0.001,
    batchSize: 32,
    epochs: 10,
    optimizer: 'adam'
  },
  steps: Array.from({ length: 5 }).map((_, stepIndex) => ({
    id: `step-${stepIndex + 1}`,
    label: `Step ${stepIndex + 1}`,
    description: `Description for step ${stepIndex + 1}`,
    status: stepIndex < Math.floor(Math.random() * 6) 
      ? 'completed' 
      : stepIndex === Math.floor(Math.random() * 6) 
        ? Math.random() > 0.8 ? 'failed' : 'in-progress' 
        : 'pending',
    estimatedTimeRemaining: '10 minutes',
    completedAt: stepIndex < Math.floor(Math.random() * 6) ? new Date().toISOString() : undefined
  })),
  resources: {
    cpu: '4 cores',
    memory: '8 GB',
    gpu: Math.random() > 0.5 ? '1 NVIDIA RTX 3080' : undefined
  }
}));

const filterExperiments = (experiments: Experiment[], filter: ExperimentFilter) => {
  return experiments.filter(exp => {
    // Status filter
    if (filter.status && filter.status.length > 0 && !filter.status.includes(exp.status)) {
      return false;
    }
    
    // Date filter
    if (filter.dateFrom && new Date(exp.createdAt) < filter.dateFrom) {
      return false;
    }
    if (filter.dateTo) {
      const dateTo = new Date(filter.dateTo);
      dateTo.setHours(23, 59, 59, 999);
      if (new Date(exp.createdAt) > dateTo) {
        return false;
      }
    }
    
    // Tags filter
    if (filter.tags && filter.tags.length > 0) {
      if (!filter.tags.some(tag => exp.tags.includes(tag))) {
        return false;
      }
    }
    
    // Search query
    if (filter.searchQuery && filter.searchQuery.trim() !== '') {
      const query = filter.searchQuery.toLowerCase();
      return (
        exp.title.toLowerCase().includes(query) ||
        exp.description.toLowerCase().includes(query) ||
        exp.tags.some(tag => tag.toLowerCase().includes(query))
      );
    }
    
    return true;
  });
};

// API functions
export const experimentService = {
  // Get all experiments with filtering, pagination, and sorting
  getExperiments: async (filter: ExperimentFilter = {}) => {
    try {
      // In real implementation, this would be an API call
      // return axios.get(`${API_URL}/experiments`, { params: filter })
      
      // Mock implementation
      let filteredExperiments = filterExperiments(mockExperiments, filter);
      
      // Sorting
      if (filter.sortBy) {
        filteredExperiments = filteredExperiments.sort((a: any, b: any) => {
          const aValue = a[filter.sortBy!];
          const bValue = b[filter.sortBy!];
          
          if (filter.sortDirection === 'desc') {
            return aValue > bValue ? -1 : aValue < bValue ? 1 : 0;
          }
          return aValue < bValue ? -1 : aValue > bValue ? 1 : 0;
        });
      } else {
        // Default sort by createdAt desc
        filteredExperiments = filteredExperiments.sort((a, b) => 
          new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
        );
      }
      
      // Pagination
      const page = filter.page || 1;
      const limit = filter.limit || 10;
      const startIndex = (page - 1) * limit;
      const endIndex = page * limit;
      const paginatedExperiments = filteredExperiments.slice(startIndex, endIndex);
      
      return {
        data: {
          experiments: paginatedExperiments,
          totalCount: filteredExperiments.length,
          totalPages: Math.ceil(filteredExperiments.length / limit),
          currentPage: page
        }
      };
    } catch (error) {
      throw error;
    }
  },
  
  // Get a single experiment by ID
  getExperiment: async (id: string) => {
    try {
      // In real implementation, this would be an API call
      // return axios.get(`${API_URL}/experiments/${id}`)
      
      // Mock implementation
      const experiment = mockExperiments.find(exp => exp.id === id);
      if (!experiment) {
        throw new Error('Experiment not found');
      }
      return { data: experiment };
    } catch (error) {
      throw error;
    }
  },
  
  // Create a new experiment
  createExperiment: async (data: CreateExperimentData) => {
    try {
      // In real implementation, this would be an API call
      // return axios.post(`${API_URL}/experiments`, data)
      
      // Mock implementation
      const newExperiment: Experiment = {
        id: `exp-${mockExperiments.length + 1}`,
        title: data.title,
        description: data.description,
        status: 'queued',
        progress: 0,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        userId: 'user-1', // This would come from auth
        tags: data.tags,
        parameters: data.parameters,
        paperUrl: data.paperUrl,
        steps: [
          {
            id: 'step-1',
            label: 'Initialization',
            description: 'Setting up the experiment environment',
            status: 'pending'
          },
          {
            id: 'step-2',
            label: 'Data Processing',
            description: 'Processing and preparing data for the experiment',
            status: 'pending'
          },
          {
            id: 'step-3',
            label: 'Model Training',
            description: 'Training the model with specified parameters',
            status: 'pending'
          },
          {
            id: 'step-4',
            label: 'Evaluation',
            description: 'Evaluating model performance',
            status: 'pending'
          },
          {
            id: 'step-5',
            label: 'Results Analysis',
            description: 'Analyzing results and generating reports',
            status: 'pending'
          }
        ],
        resources: {
          cpu: '4 cores',
          memory: '8 GB'
        }
      };
      
      mockExperiments.push(newExperiment);
      return { data: newExperiment };
    } catch (error) {
      throw error;
    }
  },
  
  // Update an experiment
  updateExperiment: async (id: string, data: Partial<Experiment>) => {
    try {
      // In real implementation, this would be an API call
      // return axios.put(`${API_URL}/experiments/${id}`, data)
      
      // Mock implementation
      const experimentIndex = mockExperiments.findIndex(exp => exp.id === id);
      if (experimentIndex === -1) {
        throw new Error('Experiment not found');
      }
      
      mockExperiments[experimentIndex] = {
        ...mockExperiments[experimentIndex],
        ...data,
        updatedAt: new Date().toISOString()
      };
      
      return { data: mockExperiments[experimentIndex] };
    } catch (error) {
      throw error;
    }
  },
  
  // Delete an experiment
  deleteExperiment: async (id: string) => {
    try {
      // In real implementation, this would be an API call
      // return axios.delete(`${API_URL}/experiments/${id}`)
      
      // Mock implementation
      const experimentIndex = mockExperiments.findIndex(exp => exp.id === id);
      if (experimentIndex === -1) {
        throw new Error('Experiment not found');
      }
      
      const deletedExperiment = mockExperiments[experimentIndex];
      mockExperiments = mockExperiments.filter(exp => exp.id !== id);
      
      return { data: deletedExperiment };
    } catch (error) {
      throw error;
    }
  },
  
  // Start an experiment
  startExperiment: async (id: string) => {
    try {
      // In real implementation, this would be an API call
      // return axios.post(`${API_URL}/experiments/${id}/start`)
      
      // Mock implementation
      const experimentIndex = mockExperiments.findIndex(exp => exp.id === id);
      if (experimentIndex === -1) {
        throw new Error('Experiment not found');
      }
      
      mockExperiments[experimentIndex] = {
        ...mockExperiments[experimentIndex],
        status: 'running',
        startTime: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      };
      
      if (mockExperiments[experimentIndex].steps && mockExperiments[experimentIndex].steps!.length > 0) {
        mockExperiments[experimentIndex].steps![0].status = 'in-progress';
      }
      
      return { data: mockExperiments[experimentIndex] };
    } catch (error) {
      throw error;
    }
  },
  
  // Pause an experiment
  pauseExperiment: async (id: string) => {
    try {
      // In real implementation, this would be an API call
      // return axios.post(`${API_URL}/experiments/${id}/pause`)
      
      // Mock implementation
      const experimentIndex = mockExperiments.findIndex(exp => exp.id === id);
      if (experimentIndex === -1) {
        throw new Error('Experiment not found');
      }
      
      mockExperiments[experimentIndex] = {
        ...mockExperiments[experimentIndex],
        status: 'paused',
        updatedAt: new Date().toISOString()
      };
      
      return { data: mockExperiments[experimentIndex] };
    } catch (error) {
      throw error;
    }
  },
  
  // Stop an experiment
  stopExperiment: async (id: string) => {
    try {
      // In real implementation, this would be an API call
      // return axios.post(`${API_URL}/experiments/${id}/stop`)
      
      // Mock implementation
      const experimentIndex = mockExperiments.findIndex(exp => exp.id === id);
      if (experimentIndex === -1) {
        throw new Error('Experiment not found');
      }
      
      mockExperiments[experimentIndex] = {
        ...mockExperiments[experimentIndex],
        status: 'completed',
        progress: 100,
        endTime: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      };
      
      if (mockExperiments[experimentIndex].steps) {
        mockExperiments[experimentIndex].steps = mockExperiments[experimentIndex].steps!.map(step => ({
          ...step,
          status: 'completed',
          completedAt: new Date().toISOString()
        }));
      }
      
      return { data: mockExperiments[experimentIndex] };
    } catch (error) {
      throw error;
    }
  },
  
  // Get all available tags
  getAllTags: async () => {
    try {
      // In real implementation, this would be an API call
      // return axios.get(`${API_URL}/experiments/tags`)
      
      // Mock implementation
      const allTags = new Set<string>();
      mockExperiments.forEach(exp => {
        exp.tags.forEach(tag => allTags.add(tag));
      });
      
      return { data: Array.from(allTags) };
    } catch (error) {
      throw error;
    }
  }
};

export default experimentService; 