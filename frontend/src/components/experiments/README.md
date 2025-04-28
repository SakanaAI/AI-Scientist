# Experiments Components for AI-Scientist

This directory contains components for the Experiments section of the AI-Scientist platform.

## Components

### ExperimentCard

A card component for displaying experiment information in a list view. It shows the experiment title, description, status, progress, and provides action buttons to start, pause, or stop the experiment.

**Props:**
- `experiment`: Experiment object with details
- `onStart`: Optional callback for starting an experiment
- `onPause`: Optional callback for pausing an experiment
- `onStop`: Optional callback for stopping an experiment

### ExperimentFilter

A component for filtering experiments by status, tags, date range, and search query.

**Props:**
- `onFilterChange`: Callback that receives the updated filters
- `availableTags`: Array of available tags for filtering

### ExperimentProgressTracker

A component for visualizing the progress of an experiment with a step-by-step tracker.

**Props:**
- `steps`: Array of step objects with status and details
- `currentStep`: The current active step index
- `overallProgress`: Overall percentage of experiment completion

### ExperimentLogViewer

A component for displaying and filtering experiment logs in real-time.

**Props:**
- `logs`: Array of log entries to display
- `loading`: Boolean indicating if logs are being loaded
- `isPolling`: Boolean indicating if real-time polling is active
- `onRefresh`: Callback to manually refresh logs
- `onClear`: Callback to clear all logs
- `onStartPolling`: Callback to start real-time polling
- `onStopPolling`: Callback to stop real-time polling

## Usage

```tsx
// Example usage of ExperimentCard
import { ExperimentCard } from '../../components/experiments';

const ExperimentsList = () => {
  const handleStart = (id: string) => {
    // Logic to start experiment
  };
  
  return (
    <ExperimentCard 
      experiment={experimentData}
      onStart={handleStart}
    />
  );
};

// Example usage of ExperimentFilter
import { ExperimentFilter } from '../../components/experiments';

const ExperimentsList = () => {
  const handleFilterChange = (filters) => {
    // Apply filters to experiment list
  };
  
  return (
    <ExperimentFilter
      onFilterChange={handleFilterChange}
      availableTags={['ML', 'Neural Networks', 'Research']}
    />
  );
};

// Example usage of ExperimentProgressTracker
import { ExperimentProgressTracker } from '../../components/experiments';

const ExperimentDetail = () => {
  return (
    <ExperimentProgressTracker
      steps={experimentSteps}
      currentStep={2}
      overallProgress={65}
    />
  );
};

// Example usage of ExperimentLogViewer
import { ExperimentLogViewer } from '../../components/experiments';
import { useExperimentLogs } from '../../hooks';

const ExperimentDetail = () => {
  const { 
    logs, 
    loading, 
    isPolling, 
    fetchLogs, 
    clearLogs, 
    startPolling, 
    stopPolling 
  } = useExperimentLogs({
    experimentId: 'exp-123'
  });
  
  return (
    <ExperimentLogViewer
      logs={logs}
      loading={loading}
      isPolling={isPolling}
      onRefresh={fetchLogs}
      onClear={clearLogs}
      onStartPolling={startPolling}
      onStopPolling={stopPolling}
    />
  );
};
```

## Data Models

The components expect specific data structures:

```typescript
// Experiment object structure
interface Experiment {
  id: string;
  title: string;
  description: string;
  status: 'running' | 'completed' | 'failed' | 'paused' | 'queued';
  progress: number;
  createdAt: string;
  updatedAt: string;
  tags: string[];
  // ...other fields
}

// Experiment step structure
interface ExperimentStep {
  id: string;
  label: string;
  description: string;
  status: 'completed' | 'in-progress' | 'pending' | 'failed';
  estimatedTimeRemaining?: string;
  completedAt?: string;
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

## Related Hooks

These components are designed to work with the following hooks:

- `useExperiments`: For managing experiment data and operations
- `useExperimentLogs`: For managing experiment logs and real-time updates 