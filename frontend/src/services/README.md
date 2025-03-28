# API Services for AI-Scientist

This directory contains service modules that handle API communication between the frontend and backend of the AI-Scientist platform.

## Services

### experimentService

Handles all experiment-related API operations.

**Methods:**
- `getExperiments(filter)`: Fetch experiments with filtering, pagination, and sorting
- `getExperiment(id)`: Fetch a single experiment by ID
- `createExperiment(data)`: Create a new experiment
- `updateExperiment(id, data)`: Update an existing experiment
- `deleteExperiment(id)`: Delete an experiment
- `startExperiment(id)`: Start an experiment
- `pauseExperiment(id)`: Pause an experiment
- `stopExperiment(id)`: Stop an experiment
- `getAllTags()`: Get all available tags

**Types:**
```typescript
interface Experiment {
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

interface ExperimentFilter {
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

interface CreateExperimentData {
  title: string;
  description: string;
  parameters: Record<string, any>;
  tags: string[];
  paperUrl?: string;
}
```

### authService

Handles authentication-related API operations.

**Methods:**
- `login(email, password)`: Authenticate a user
- `register(userData)`: Register a new user
- `logout()`: Log out the current user
- `refreshToken()`: Refresh the authentication token
- `getCurrentUser()`: Get the current authenticated user

## Usage Example

```typescript
import experimentService from '../services/experimentService';

// Fetch experiments with filtering
const fetchExperiments = async () => {
  try {
    const response = await experimentService.getExperiments({
      status: ['running', 'queued'],
      tags: ['ML', 'Research'],
      page: 1,
      limit: 10,
      sortBy: 'createdAt',
      sortDirection: 'desc'
    });
    
    return response.data;
  } catch (error) {
    console.error('Error fetching experiments:', error);
    throw error;
  }
};

// Create a new experiment
const createExperiment = async (experimentData) => {
  try {
    const response = await experimentService.createExperiment({
      title: 'New Experiment',
      description: 'This is a test experiment',
      parameters: {
        learningRate: 0.001,
        batchSize: 32
      },
      tags: ['Research', 'Test']
    });
    
    return response.data;
  } catch (error) {
    console.error('Error creating experiment:', error);
    throw error;
  }
};
```

## Implementation Notes

The current implementation uses mock data for demonstration purposes. In a production environment, these services would make actual API calls to the backend server.

To configure the API URL, you can set the `REACT_APP_API_URL` environment variable, or it will default to `http://localhost:8000/api`.

Example of actual API call implementation:

```typescript
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

export const experimentService = {
  getExperiments: async (filter: ExperimentFilter = {}) => {
    return axios.get(`${API_URL}/experiments`, { params: filter });
  },
  
  getExperiment: async (id: string) => {
    return axios.get(`${API_URL}/experiments/${id}`);
  },
  
  // ... other methods
};
``` 