import { useState, useEffect, useCallback, useRef } from 'react';

// Types
export interface LogEntry {
  timestamp: string;
  level: 'info' | 'warning' | 'error' | 'debug';
  message: string;
  source?: string;
  data?: any;
}

// For demo, we'll use a mock implementation
const mockFetchLogs = async (experimentId: string, limit: number = 100, after?: string): Promise<LogEntry[]> => {
  // In a real implementation, this would make an API call
  // Something like: return api.get(`/experiments/${experimentId}/logs?limit=${limit}&after=${after}`)
  
  // For demo, generate some mock logs
  const levels: Array<'info' | 'warning' | 'error' | 'debug'> = ['info', 'warning', 'error', 'debug'];
  const sources = ['system', 'model', 'data', 'evaluation'];
  
  return Array.from({ length: 20 }).map((_, i) => {
    const now = new Date();
    now.setSeconds(now.getSeconds() - (20 - i) * 30); // Spread logs over the last 10 minutes
    
    return {
      timestamp: now.toISOString(),
      level: levels[Math.floor(Math.random() * levels.length)],
      message: `Log message #${i + 1} for experiment ${experimentId}`,
      source: sources[Math.floor(Math.random() * sources.length)],
      data: Math.random() > 0.7 ? { detail: `Additional data for log ${i + 1}` } : undefined
    };
  });
};

interface UseExperimentLogsProps {
  experimentId: string;
  pollingInterval?: number; // in milliseconds
  limit?: number;
}

export const useExperimentLogs = ({
  experimentId,
  pollingInterval = 5000, // Default to 5 seconds
  limit = 100
}: UseExperimentLogsProps) => {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isPolling, setIsPolling] = useState(true);
  
  const lastTimestampRef = useRef<string | undefined>(undefined);
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  
  const fetchLogs = useCallback(async (isInitial: boolean = false) => {
    if (!experimentId) return;
    
    try {
      setLoading(true);
      
      const newLogs = await mockFetchLogs(
        experimentId,
        limit,
        isInitial ? undefined : lastTimestampRef.current
      );
      
      if (newLogs.length > 0) {
        if (isInitial) {
          setLogs(newLogs);
        } else {
          setLogs(prev => [...prev, ...newLogs]);
        }
        
        // Update the last timestamp for next fetch
        if (newLogs.length > 0) {
          const latestLog = newLogs[newLogs.length - 1];
          lastTimestampRef.current = latestLog.timestamp;
        }
      }
      
      setError(null);
    } catch (err) {
      console.error('Error fetching logs:', err);
      setError('Failed to load logs. Please try again.');
    } finally {
      setLoading(false);
    }
  }, [experimentId, limit]);
  
  const startPolling = useCallback(() => {
    if (!isPolling) {
      setIsPolling(true);
    }
  }, [isPolling]);
  
  const stopPolling = useCallback(() => {
    if (isPolling) {
      setIsPolling(false);
    }
  }, [isPolling]);
  
  const clearLogs = useCallback(() => {
    setLogs([]);
    lastTimestampRef.current = undefined;
  }, []);
  
  // Initial fetch and polling setup
  useEffect(() => {
    fetchLogs(true);
    
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
      }
    };
  }, [fetchLogs]);
  
  // Manage polling interval
  useEffect(() => {
    if (isPolling) {
      pollingIntervalRef.current = setInterval(() => {
        fetchLogs(false);
      }, pollingInterval);
    } else if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
    }
    
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
      }
    };
  }, [isPolling, pollingInterval, fetchLogs]);
  
  // Filter logs by level
  const filterLogsByLevel = useCallback((level: 'info' | 'warning' | 'error' | 'debug') => {
    return logs.filter(log => log.level === level);
  }, [logs]);
  
  // Filter logs by source
  const filterLogsBySource = useCallback((source: string) => {
    return logs.filter(log => log.source === source);
  }, [logs]);
  
  return {
    logs,
    loading,
    error,
    isPolling,
    startPolling,
    stopPolling,
    clearLogs,
    filterLogsByLevel,
    filterLogsBySource
  };
};

export default useExperimentLogs; 