import React, { useState } from 'react';
import { 
  Box, 
  TextField, 
  MenuItem, 
  Chip,
  FormControl,
  InputLabel,
  Select,
  OutlinedInput,
  SelectChangeEvent,
  Grid,
  Button,
  Typography,
  IconButton,
  Collapse
} from '@mui/material';
import { DatePicker } from '@mui/x-date-pickers/DatePicker';
import { FilterList, Clear } from '@mui/icons-material';

interface ExperimentFilterProps {
  onFilterChange: (filters: {
    status: string[];
    dateFrom: Date | null;
    dateTo: Date | null;
    tags: string[];
    searchQuery: string;
  }) => void;
  availableTags: string[];
}

const statusOptions = [
  { value: 'running', label: 'Running' },
  { value: 'completed', label: 'Completed' },
  { value: 'failed', label: 'Failed' },
  { value: 'paused', label: 'Paused' },
  { value: 'queued', label: 'Queued' }
];

const ExperimentFilter: React.FC<ExperimentFilterProps> = ({ onFilterChange, availableTags }) => {
  const [status, setStatus] = useState<string[]>([]);
  const [dateFrom, setDateFrom] = useState<Date | null>(null);
  const [dateTo, setDateTo] = useState<Date | null>(null);
  const [tags, setTags] = useState<string[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [showFilters, setShowFilters] = useState(false);
  
  const handleStatusChange = (event: SelectChangeEvent<typeof status>) => {
    const value = event.target.value;
    setStatus(typeof value === 'string' ? value.split(',') : value);
    applyFilters({ status: typeof value === 'string' ? value.split(',') : value });
  };
  
  const handleTagsChange = (event: SelectChangeEvent<typeof tags>) => {
    const value = event.target.value;
    setTags(typeof value === 'string' ? value.split(',') : value);
    applyFilters({ tags: typeof value === 'string' ? value.split(',') : value });
  };
  
  const handleSearchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(event.target.value);
    applyFilters({ searchQuery: event.target.value });
  };
  
  const handleDateFromChange = (date: Date | null) => {
    setDateFrom(date);
    applyFilters({ dateFrom: date });
  };
  
  const handleDateToChange = (date: Date | null) => {
    setDateTo(date);
    applyFilters({ dateTo: date });
  };
  
  const applyFilters = (updatedFilters: Partial<{
    status: string[];
    dateFrom: Date | null;
    dateTo: Date | null;
    tags: string[];
    searchQuery: string;
  }>) => {
    onFilterChange({
      status: updatedFilters.status !== undefined ? updatedFilters.status : status,
      dateFrom: updatedFilters.dateFrom !== undefined ? updatedFilters.dateFrom : dateFrom,
      dateTo: updatedFilters.dateTo !== undefined ? updatedFilters.dateTo : dateTo,
      tags: updatedFilters.tags !== undefined ? updatedFilters.tags : tags,
      searchQuery: updatedFilters.searchQuery !== undefined ? updatedFilters.searchQuery : searchQuery
    });
  };
  
  const clearFilters = () => {
    setStatus([]);
    setDateFrom(null);
    setDateTo(null);
    setTags([]);
    setSearchQuery('');
    onFilterChange({
      status: [],
      dateFrom: null,
      dateTo: null,
      tags: [],
      searchQuery: ''
    });
  };
  
  const toggleFilters = () => {
    setShowFilters(!showFilters);
  };
  
  const hasActiveFilters = () => {
    return status.length > 0 || dateFrom !== null || dateTo !== null || tags.length > 0 || searchQuery !== '';
  };
  
  return (
    <Box sx={{ mb: 3 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <TextField
          placeholder="Search experiments"
          variant="outlined"
          fullWidth
          size="small"
          value={searchQuery}
          onChange={handleSearchChange}
          sx={{ mr: 2, flexGrow: 1 }}
        />
        <Box display="flex" alignItems="center">
          {hasActiveFilters() && (
            <Button 
              size="small" 
              onClick={clearFilters} 
              startIcon={<Clear />}
              sx={{ mr: 1 }}
            >
              Clear
            </Button>
          )}
          <Button 
            variant={showFilters ? "contained" : "outlined"} 
            color="primary" 
            onClick={toggleFilters}
            startIcon={<FilterList />}
            size="small"
          >
            Filters
          </Button>
        </Box>
      </Box>
      
      <Collapse in={showFilters}>
        <Grid container spacing={2} sx={{ mb: 2 }}>
          <Grid item xs={12} sm={6} md={3}>
            <FormControl fullWidth size="small">
              <InputLabel id="status-filter-label">Status</InputLabel>
              <Select
                labelId="status-filter-label"
                multiple
                value={status}
                onChange={handleStatusChange}
                input={<OutlinedInput label="Status" />}
                renderValue={(selected) => (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {selected.map((value) => (
                      <Chip 
                        key={value} 
                        label={statusOptions.find(option => option.value === value)?.label || value} 
                        size="small" 
                      />
                    ))}
                  </Box>
                )}
              >
                {statusOptions.map((option) => (
                  <MenuItem key={option.value} value={option.value}>
                    {option.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <FormControl fullWidth size="small">
              <InputLabel id="tags-filter-label">Tags</InputLabel>
              <Select
                labelId="tags-filter-label"
                multiple
                value={tags}
                onChange={handleTagsChange}
                input={<OutlinedInput label="Tags" />}
                renderValue={(selected) => (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {selected.map((value) => (
                      <Chip key={value} label={value} size="small" />
                    ))}
                  </Box>
                )}
              >
                {availableTags.map((tag) => (
                  <MenuItem key={tag} value={tag}>
                    {tag}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <DatePicker
              label="From Date"
              value={dateFrom}
              onChange={handleDateFromChange}
              slotProps={{ textField: { size: 'small', fullWidth: true } }}
            />
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <DatePicker
              label="To Date"
              value={dateTo}
              onChange={handleDateToChange}
              slotProps={{ textField: { size: 'small', fullWidth: true } }}
            />
          </Grid>
        </Grid>
      </Collapse>
      
      {hasActiveFilters() && (
        <Box display="flex" flexWrap="wrap" gap={1} mt={1}>
          <Typography variant="body2" color="text.secondary" sx={{ mr: 1 }}>
            Active filters:
          </Typography>
          
          {status.map((s) => (
            <Chip 
              key={`status-${s}`}
              label={`Status: ${statusOptions.find(option => option.value === s)?.label || s}`}
              size="small"
              onDelete={() => {
                const newStatus = status.filter(item => item !== s);
                setStatus(newStatus);
                applyFilters({ status: newStatus });
              }}
            />
          ))}
          
          {tags.map((tag) => (
            <Chip 
              key={`tag-${tag}`}
              label={`Tag: ${tag}`}
              size="small"
              onDelete={() => {
                const newTags = tags.filter(t => t !== tag);
                setTags(newTags);
                applyFilters({ tags: newTags });
              }}
            />
          ))}
          
          {dateFrom && (
            <Chip 
              label={`From: ${dateFrom.toLocaleDateString()}`}
              size="small"
              onDelete={() => {
                setDateFrom(null);
                applyFilters({ dateFrom: null });
              }}
            />
          )}
          
          {dateTo && (
            <Chip 
              label={`To: ${dateTo.toLocaleDateString()}`}
              size="small"
              onDelete={() => {
                setDateTo(null);
                applyFilters({ dateTo: null });
              }}
            />
          )}
          
          {searchQuery && (
            <Chip 
              label={`Search: ${searchQuery}`}
              size="small"
              onDelete={() => {
                setSearchQuery('');
                applyFilters({ searchQuery: '' });
              }}
            />
          )}
        </Box>
      )}
    </Box>
  );
};

export default ExperimentFilter; 