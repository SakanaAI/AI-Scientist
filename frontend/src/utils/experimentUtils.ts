import { Experiment } from '../services/experimentService';

/**
 * Format a timestamp into a readable date/time string
 * @param timestamp ISO timestamp string
 * @param includeTime Whether to include the time in the output
 */
export const formatDate = (timestamp: string, includeTime: boolean = false): string => {
  if (!timestamp) return 'N/A';
  
  const date = new Date(timestamp);
  const options: Intl.DateTimeFormatOptions = {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    ...(includeTime ? { hour: '2-digit', minute: '2-digit' } : {})
  };
  
  return date.toLocaleDateString(undefined, options);
};

/**
 * Format experiment status for display
 * @param status Experiment status string
 */
export const formatStatus = (status: string): string => {
  if (!status) return 'Unknown';
  
  return status.charAt(0).toUpperCase() + status.slice(1);
};

/**
 * Format a duration in seconds to a human-readable string
 * @param seconds Duration in seconds
 */
export const formatDuration = (seconds: number): string => {
  if (!seconds || isNaN(seconds)) return 'N/A';
  
  if (seconds < 60) {
    return `${Math.round(seconds)} seconds`;
  } else if (seconds < 3600) {
    const minutes = Math.floor(seconds / 60);
    return `${minutes} minute${minutes !== 1 ? 's' : ''}`;
  } else if (seconds < 86400) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours} hour${hours !== 1 ? 's' : ''}${minutes > 0 ? `, ${minutes} minute${minutes !== 1 ? 's' : ''}` : ''}`;
  } else {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    return `${days} day${days !== 1 ? 's' : ''}${hours > 0 ? `, ${hours} hour${hours !== 1 ? 's' : ''}` : ''}`;
  }
};

/**
 * Calculate the duration between two ISO timestamps in seconds
 * @param startTime Start time ISO string
 * @param endTime End time ISO string (defaults to now if not provided)
 */
export const calculateDuration = (startTime: string, endTime?: string): number => {
  if (!startTime) return 0;
  
  const start = new Date(startTime).getTime();
  const end = endTime ? new Date(endTime).getTime() : Date.now();
  
  return Math.round((end - start) / 1000);
};

/**
 * Calculate the estimated completion time based on progress and elapsed time
 * @param experiment Experiment object
 */
export const calculateEstimatedCompletion = (experiment: Experiment): string => {
  if (!experiment.startTime || experiment.status === 'completed' || experiment.status === 'failed') {
    return 'N/A';
  }
  
  const elapsedSeconds = calculateDuration(experiment.startTime);
  const progress = experiment.progress;
  
  if (progress <= 0) return 'Calculating...';
  
  const totalEstimatedSeconds = (elapsedSeconds / progress) * 100;
  const remainingSeconds = totalEstimatedSeconds - elapsedSeconds;
  
  if (remainingSeconds <= 0) return 'Almost done';
  
  return formatDuration(remainingSeconds);
};

/**
 * Get a color for a specific experiment status
 * @param status Experiment status
 */
export const getStatusColor = (status: string): string => {
  switch (status) {
    case 'running':
      return 'primary';
    case 'completed':
      return 'success';
    case 'failed':
      return 'error';
    case 'paused':
      return 'warning';
    case 'queued':
      return 'info';
    default:
      return 'default';
  }
};

/**
 * Generate a simplified summary of experiment parameters
 * @param parameters Experiment parameters object
 * @param maxLength Maximum length of generated summary
 */
export const generateParameterSummary = (parameters: Record<string, any>, maxLength: number = 100): string => {
  if (!parameters || Object.keys(parameters).length === 0) {
    return 'No parameters';
  }
  
  const summary = Object.entries(parameters)
    .map(([key, value]) => {
      // Handle basic types directly
      if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
        return `${key}: ${value}`;
      }
      // For arrays, show length
      if (Array.isArray(value)) {
        return `${key}: [${value.length} items]`;
      }
      // For objects, show a summary
      if (typeof value === 'object' && value !== null) {
        return `${key}: {${Object.keys(value).length} properties}`;
      }
      // For null values
      if (value === null) {
        return `${key}: null`;
      }
      // For other types
      return `${key}: ${typeof value}`;
    })
    .join(', ');
    
  if (summary.length <= maxLength) {
    return summary;
  }
  
  return summary.substring(0, maxLength - 3) + '...';
};

/**
 * Generate experiment resource usage summary
 * @param resources Experiment resources object
 */
export const generateResourcesSummary = (resources?: { cpu: string; memory: string; gpu?: string }): string => {
  if (!resources) return 'N/A';
  
  let summary = `CPU: ${resources.cpu}, Memory: ${resources.memory}`;
  if (resources.gpu) {
    summary += `, GPU: ${resources.gpu}`;
  }
  
  return summary;
};

/**
 * Get the overall status of an experiment's steps
 * @param steps Array of experiment steps
 */
export const getStepsStatus = (steps?: Array<{ status: string }>): { 
  completed: number; 
  total: number; 
  hasErrors: boolean; 
} => {
  if (!steps || steps.length === 0) {
    return { completed: 0, total: 0, hasErrors: false };
  }
  
  const completed = steps.filter(step => step.status === 'completed').length;
  const hasErrors = steps.some(step => step.status === 'failed');
  
  return {
    completed,
    total: steps.length,
    hasErrors
  };
}; 