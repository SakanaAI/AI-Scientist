# Custom Hooks for AI-Scientist

This directory contains custom React hooks that provide reusable logic for the AI-Scientist platform.

## Hooks

### useExperiments

A hook for managing experiment data and operations.

```typescript
const {
  // Data
  experiments,         // Array of experiments
  experiment,          // Single experiment details
  loading,             // Loading state
  error,               // Error state
  totalCount,          // Total count of experiments
  totalPages,          // Total pages for pagination
  currentPage,         // Current page number
  limit,               // Page size limit
  filters,             // Current filter state
  availableTags,       // Available tags for filtering
  
  // Methods
  fetchExperiments,    // Fetch experiments with optional filters
  fetchExperiment,     // Fetch a single experiment by ID
  createExperiment,    // Create a new experiment
  updateExperiment,    // Update an existing experiment
  deleteExperiment,    // Delete an experiment
  startExperiment,     // Start an experiment
  pauseExperiment,     // Pause an experiment
  stopExperiment,      // Stop an experiment
  fetchTags,           // Fetch all available tags
  handleFilterChange,  // Update filters
  handlePageChange     // Change the current page
} = useExperiments(props);
```

**Props:**
- `initialFilters`: Optional initial filter state

### useExperimentLogs

A hook for managing and displaying experiment logs in real-time.

```typescript
const {
  logs,            // Array of log entries
  loading,         // Loading state
  error,           // Error state
  isPolling,       // Whether real-time polling is active
  startPolling,    // Start real-time polling
  stopPolling,     // Stop real-time polling
  clearLogs,       // Clear all logs
  filterLogsByLevel,  // Filter logs by level
  filterLogsBySource  // Filter logs by source
} = useExperimentLogs({
  experimentId,     // ID of the experiment
  pollingInterval,  // Optional polling interval in milliseconds (default: 5000)
  limit             // Optional limit of logs to fetch (default: 100)
});
```

## Data Models

```typescript
// Experiment filter
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

// Log entry structure
interface LogEntry {
  timestamp: string;
  level: 'info' | 'warning' | 'error' | 'debug';
  message: string;
  source?: string;
  data?: any;
}
```

## Usage Examples

### useExperiments Example

```tsx
import { useExperiments } from '../hooks';

const ExperimentsList = () => {
  const {
    experiments,
    loading,
    error,
    totalPages,
    currentPage,
    fetchExperiments,
    handleFilterChange,
    handlePageChange
  } = useExperiments();
  
  // Initial fetch
  useEffect(() => {
    fetchExperiments();
  }, [fetchExperiments]);
  
  // Handle filter changes
  const onFilterChange = (newFilters) => {
    handleFilterChange(newFilters);
  };
  
  // Handle pagination
  const onPageChange = (page) => {
    handlePageChange(page);
  };
  
  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error}</div>;
  
  return (
    <div>
      {/* Render experiments */}
    </div>
  );
};
```

### useExperimentLogs Example

```tsx
import { useExperimentLogs } from '../hooks';

const ExperimentLogs = ({ experimentId }) => {
  const {
    logs,
    loading,
    error,
    isPolling,
    startPolling,
    stopPolling,
    clearLogs
  } = useExperimentLogs({
    experimentId,
    pollingInterval: 3000 // Poll every 3 seconds
  });
  
  const togglePolling = () => {
    if (isPolling) {
      stopPolling();
    } else {
      startPolling();
    }
  };
  
  return (
    <div>
      <button onClick={togglePolling}>
        {isPolling ? 'Pause Updates' : 'Resume Updates'}
      </button>
      <button onClick={clearLogs}>Clear Logs</button>
      
      {loading && <div>Loading...</div>}
      {error && <div>Error: {error}</div>}
      
      <div>
        {logs.map((log, index) => (
          <div key={index}>
            {log.timestamp} - {log.level}: {log.message}
          </div>
        ))}
      </div>
    </div>
  );
};
``` 