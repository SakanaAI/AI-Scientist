import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  Chip,
  Button,
  TextField,
  InputAdornment,
  IconButton,
  Menu,
  MenuItem,
  Card,
  CardContent,
  Typography,
  Grid,
  Divider,
  FormControl,
  InputLabel,
  Select,
  SelectChangeEvent
} from '@mui/material';
import {
  Add as AddIcon,
  Search as SearchIcon,
  FilterList as FilterListIcon,
  Science as ScienceIcon,
  MoreVert as MoreVertIcon,
  CheckCircle,
  Cancel,
  Pending,
  Delete as DeleteIcon
} from '@mui/icons-material';
import PageHeader from '../../components/common/PageHeader';

// Mock data for experiments
const experiments = [
  { id: '1', name: 'Transformer Self-Attention Patterns', template: 'nanoGPT', status: 'completed', date: '2023-11-02', creator: 'AI-Scientist', ideas: 5 },
  { id: '2', name: 'Efficient 2D Diffusion Model Training', template: '2d_diffusion', status: 'in_progress', date: '2023-11-01', creator: 'AI-Scientist', ideas: 3 },
  { id: '3', name: 'Effects of Learning Rate on Grokking', template: 'grokking', status: 'planned', date: '2023-10-28', creator: 'AI-Scientist', ideas: 2 },
  { id: '4', name: 'Analysis of RLHF Training', template: 'custom', status: 'completed', date: '2023-10-25', creator: 'AI-Scientist', ideas: 4 },
  { id: '5', name: 'Scaling Laws for Vision Models', template: 'nanoGPT', status: 'cancelled', date: '2023-10-20', creator: 'AI-Scientist', ideas: 3 },
  { id: '6', name: 'Comparison of Diffusion Samplers', template: '2d_diffusion', status: 'completed', date: '2023-10-15', creator: 'AI-Scientist', ideas: 5 },
  { id: '7', name: 'Transformer vs CNN for Vision Tasks', template: 'custom', status: 'in_progress', date: '2023-10-12', creator: 'AI-Scientist', ideas: 2 },
  { id: '8', name: 'Fine-tuning LLMs for Scientific Tasks', template: 'nanoGPT', status: 'completed', date: '2023-10-05', creator: 'AI-Scientist', ideas: 4 },
];

const ExperimentsList: React.FC = () => {
  const navigate = useNavigate();
  
  // Pagination state
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(5);
  
  // Filter state
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [templateFilter, setTemplateFilter] = useState('all');
  
  // Context menu state
  const [menuAnchorEl, setMenuAnchorEl] = useState<null | HTMLElement>(null);
  const [selectedExperimentId, setSelectedExperimentId] = useState<string | null>(null);
  
  const handleChangePage = (event: unknown, newPage: number) => {
    setPage(newPage);
  };
  
  const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };
  
  const handleSearchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(event.target.value);
    setPage(0);
  };
  
  const handleStatusFilterChange = (event: SelectChangeEvent) => {
    setStatusFilter(event.target.value);
    setPage(0);
  };
  
  const handleTemplateFilterChange = (event: SelectChangeEvent) => {
    setTemplateFilter(event.target.value);
    setPage(0);
  };
  
  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>, experimentId: string) => {
    setMenuAnchorEl(event.currentTarget);
    setSelectedExperimentId(experimentId);
  };
  
  const handleMenuClose = () => {
    setMenuAnchorEl(null);
    setSelectedExperimentId(null);
  };
  
  const handleViewDetails = () => {
    if (selectedExperimentId) {
      navigate(`/experiments/${selectedExperimentId}`);
    }
    handleMenuClose();
  };
  
  const getStatusChip = (status: string) => {
    switch (status) {
      case 'completed':
        return <Chip icon={<CheckCircle fontSize="small" />} label="Completed" color="success" size="small" />;
      case 'in_progress':
        return <Chip icon={<Pending fontSize="small" />} label="In Progress" color="primary" size="small" />;
      case 'planned':
        return <Chip icon={<Pending fontSize="small" />} label="Planned" color="default" size="small" />;
      case 'cancelled':
        return <Chip icon={<Cancel fontSize="small" />} label="Cancelled" color="error" size="small" />;
      default:
        return <Chip label={status} color="default" size="small" />;
    }
  };
  
  // Filter experiments based on search and filters
  const filteredExperiments = experiments.filter((experiment) => {
    const matchesSearch = 
      experiment.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      experiment.template.toLowerCase().includes(searchTerm.toLowerCase());
      
    const matchesStatus = statusFilter === 'all' || experiment.status === statusFilter;
    const matchesTemplate = templateFilter === 'all' || experiment.template === templateFilter;
    
    return matchesSearch && matchesStatus && matchesTemplate;
  });
  
  // Get unique templates for the filter
  const uniqueTemplates = Array.from(new Set(experiments.map(exp => exp.template)));
  
  // Stats for the summary cards
  const completedCount = experiments.filter(exp => exp.status === 'completed').length;
  const inProgressCount = experiments.filter(exp => exp.status === 'in_progress').length;
  const plannedCount = experiments.filter(exp => exp.status === 'planned').length;
  
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
                {experiments.length}
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
                In Progress
              </Typography>
              <Typography variant="h4" component="div" color="primary.main">
                {inProgressCount}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Planned
              </Typography>
              <Typography variant="h4" component="div" color="text.secondary">
                {plannedCount}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
      
      <Paper sx={{ mb: 4 }}>
        {/* Filters */}
        <Box sx={{ p: 2, display: 'flex', flexWrap: 'wrap', gap: 2 }}>
          <TextField
            placeholder="Search experiments..."
            variant="outlined"
            size="small"
            value={searchTerm}
            onChange={handleSearchChange}
            sx={{ flex: 1, minWidth: '200px' }}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon />
                </InputAdornment>
              ),
            }}
          />
          
          <FormControl size="small" sx={{ minWidth: '150px' }}>
            <InputLabel id="status-filter-label">Status</InputLabel>
            <Select
              labelId="status-filter-label"
              id="status-filter"
              value={statusFilter}
              label="Status"
              onChange={handleStatusFilterChange}
            >
              <MenuItem value="all">All Statuses</MenuItem>
              <MenuItem value="completed">Completed</MenuItem>
              <MenuItem value="in_progress">In Progress</MenuItem>
              <MenuItem value="planned">Planned</MenuItem>
              <MenuItem value="cancelled">Cancelled</MenuItem>
            </Select>
          </FormControl>
          
          <FormControl size="small" sx={{ minWidth: '150px' }}>
            <InputLabel id="template-filter-label">Template</InputLabel>
            <Select
              labelId="template-filter-label"
              id="template-filter"
              value={templateFilter}
              label="Template"
              onChange={handleTemplateFilterChange}
            >
              <MenuItem value="all">All Templates</MenuItem>
              {uniqueTemplates.map(template => (
                <MenuItem key={template} value={template}>{template}</MenuItem>
              ))}
            </Select>
          </FormControl>
        </Box>
        
        <Divider />
        
        {/* Experiments Table */}
        <TableContainer>
          <Table sx={{ minWidth: 650 }} aria-label="experiments table">
            <TableHead>
              <TableRow>
                <TableCell>Name</TableCell>
                <TableCell>Template</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Date</TableCell>
                <TableCell>Ideas</TableCell>
                <TableCell align="right">Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {filteredExperiments
                .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                .map((experiment) => (
                  <TableRow
                    key={experiment.id}
                    sx={{ '&:last-child td, &:last-child th': { border: 0 }, cursor: 'pointer' }}
                    hover
                    onClick={() => navigate(`/experiments/${experiment.id}`)}
                  >
                    <TableCell component="th" scope="row">
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <ScienceIcon sx={{ mr: 1, color: 'primary.main' }} />
                        {experiment.name}
                      </Box>
                    </TableCell>
                    <TableCell>{experiment.template}</TableCell>
                    <TableCell>{getStatusChip(experiment.status)}</TableCell>
                    <TableCell>{experiment.date}</TableCell>
                    <TableCell>{experiment.ideas}</TableCell>
                    <TableCell align="right">
                      <IconButton 
                        aria-label="experiment actions" 
                        onClick={(e) => {
                          e.stopPropagation();
                          handleMenuOpen(e, experiment.id);
                        }}
                      >
                        <MoreVertIcon />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                ))}
              {filteredExperiments.length === 0 && (
                <TableRow>
                  <TableCell colSpan={6} align="center" sx={{ py: 3 }}>
                    <Typography variant="body1" color="text.secondary">
                      No experiments found matching your filters.
                    </Typography>
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </TableContainer>
        
        <TablePagination
          rowsPerPageOptions={[5, 10, 25]}
          component="div"
          count={filteredExperiments.length}
          rowsPerPage={rowsPerPage}
          page={page}
          onPageChange={handleChangePage}
          onRowsPerPageChange={handleChangeRowsPerPage}
        />
      </Paper>
      
      {/* Context Menu */}
      <Menu
        anchorEl={menuAnchorEl}
        open={Boolean(menuAnchorEl)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={handleViewDetails}>View Details</MenuItem>
        <MenuItem onClick={handleMenuClose}>Generate Paper</MenuItem>
        <MenuItem onClick={handleMenuClose}>Clone</MenuItem>
        <Divider />
        <MenuItem onClick={handleMenuClose} sx={{ color: 'error.main' }}>
          <DeleteIcon fontSize="small" sx={{ mr: 1 }} />
          Delete
        </MenuItem>
      </Menu>
    </Box>
  );
};

export default ExperimentsList; 