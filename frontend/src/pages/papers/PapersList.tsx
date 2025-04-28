import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  CardActions,
  Button,
  Chip,
  TextField,
  InputAdornment,
  IconButton,
  Menu,
  MenuItem,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  ListItemSecondaryAction,
  FormControl,
  InputLabel,
  Select,
  SelectChangeEvent,
  Tabs,
  Tab
} from '@mui/material';
import {
  Article,
  Search as SearchIcon,
  FilterList as FilterListIcon,
  MoreVert as MoreVertIcon,
  Description,
  CheckCircle,
  Edit,
  CloudDownload,
  Delete,
  ScienceOutlined,
  LibraryBooks,
  Visibility,
  Share
} from '@mui/icons-material';
import PageHeader from '../../components/common/PageHeader';

// Mock data for papers
const papers = [
  {
    id: '1',
    title: 'Understanding Emergent Abilities in Large Language Models',
    abstract: 'This paper explores the phenomenon of emergent abilities in large language models, where capabilities appear suddenly as model scale increases.',
    experiment: { id: '1', name: 'Transformer Self-Attention Patterns' },
    status: 'published',
    date: '2023-10-30',
    author: 'AI-Scientist',
    tags: ['LLM', 'Scaling', 'Emergent Abilities'],
    citations: 12,
    downloads: 89
  },
  {
    id: '2',
    title: 'Efficient Image Generation with Diffusion Models',
    abstract: 'This work proposes a novel approach to accelerate image generation in diffusion models while maintaining quality.',
    experiment: { id: '2', name: 'Efficient 2D Diffusion Model Training' },
    status: 'draft',
    date: '2023-10-25',
    author: 'AI-Scientist',
    tags: ['Diffusion Models', 'Image Generation', 'Efficiency'],
    citations: 0,
    downloads: 15
  },
  {
    id: '3',
    title: 'A Comparative Analysis of Training Dynamics in Grokking',
    abstract: 'We investigate the grokking phenomenon where models suddenly generalize after prolonged training, providing a theoretical framework.',
    experiment: { id: '3', name: 'Effects of Learning Rate on Grokking' },
    status: 'review',
    date: '2023-10-15',
    author: 'AI-Scientist',
    tags: ['Grokking', 'Generalization', 'Optimization'],
    citations: 3,
    downloads: 42
  },
  {
    id: '4',
    title: 'Self-Attention Patterns and Linguistic Features in Transformers',
    abstract: 'This paper analyzes how transformer attention patterns correlate with linguistic features in natural language processing tasks.',
    experiment: { id: '1', name: 'Transformer Self-Attention Patterns' },
    status: 'published',
    date: '2023-09-20',
    author: 'AI-Scientist',
    tags: ['Transformers', 'Attention', 'Linguistics'],
    citations: 18,
    downloads: 124
  },
];

// Paper view types
type ViewType = 'grid' | 'list';

const PapersList: React.FC = () => {
  const navigate = useNavigate();
  
  // State
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [viewType, setViewType] = useState<ViewType>('grid');
  const [tabValue, setTabValue] = useState(0);
  
  // Menu state
  const [menuAnchorEl, setMenuAnchorEl] = useState<null | HTMLElement>(null);
  const [selectedPaperId, setSelectedPaperId] = useState<string | null>(null);
  
  const handleSearchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(event.target.value);
  };
  
  const handleStatusFilterChange = (event: SelectChangeEvent) => {
    setStatusFilter(event.target.value);
  };
  
  const handleViewTypeChange = (newViewType: ViewType) => {
    setViewType(newViewType);
  };
  
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };
  
  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>, paperId: string) => {
    event.stopPropagation();
    setMenuAnchorEl(event.currentTarget);
    setSelectedPaperId(paperId);
  };
  
  const handleMenuClose = () => {
    setMenuAnchorEl(null);
    setSelectedPaperId(null);
  };
  
  const handlePaperClick = (paperId: string) => {
    navigate(`/papers/${paperId}`);
  };
  
  const getStatusChip = (status: string) => {
    switch (status) {
      case 'published':
        return <Chip icon={<CheckCircle fontSize="small" />} label="Published" color="success" size="small" />;
      case 'draft':
        return <Chip icon={<Edit fontSize="small" />} label="Draft" color="default" size="small" />;
      case 'review':
        return <Chip icon={<Visibility fontSize="small" />} label="In Review" color="primary" size="small" />;
      default:
        return <Chip label={status} color="default" size="small" />;
    }
  };
  
  // Filter papers based on search term and status filter
  const filteredPapers = papers.filter((paper) => {
    const matchesSearch = 
      paper.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      paper.abstract.toLowerCase().includes(searchTerm.toLowerCase()) ||
      paper.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()));
    
    const matchesStatus = statusFilter === 'all' || paper.status === statusFilter;
    const matchesTab = tabValue === 0 || 
      (tabValue === 1 && paper.status === 'published') || 
      (tabValue === 2 && paper.status === 'draft') || 
      (tabValue === 3 && paper.status === 'review');
    
    return matchesSearch && matchesStatus && matchesTab;
  });
  
  return (
    <Box>
      <PageHeader
        title="Scientific Papers"
        subtitle="Browse, edit, and generate scientific papers based on experiments"
        action={{
          label: "Generate Paper",
          icon: <Article />,
          onClick: () => navigate('/experiments')
        }}
      />
      
      {/* Stats Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Total Papers
              </Typography>
              <Typography variant="h4" component="div">
                {papers.length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Published
              </Typography>
              <Typography variant="h4" component="div" color="success.main">
                {papers.filter(p => p.status === 'published').length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Drafts
              </Typography>
              <Typography variant="h4" component="div" color="text.secondary">
                {papers.filter(p => p.status === 'draft').length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Total Citations
              </Typography>
              <Typography variant="h4" component="div" color="primary.main">
                {papers.reduce((total, paper) => total + paper.citations, 0)}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
      
      <Paper sx={{ mb: 4 }}>
        {/* Tabs */}
        <Tabs 
          value={tabValue} 
          onChange={handleTabChange}
          aria-label="paper status tabs"
          sx={{ borderBottom: 1, borderColor: 'divider' }}
        >
          <Tab icon={<LibraryBooks />} label="All Papers" />
          <Tab icon={<CheckCircle />} label="Published" />
          <Tab icon={<Edit />} label="Drafts" />
          <Tab icon={<Visibility />} label="In Review" />
        </Tabs>
        
        {/* Filters */}
        <Box sx={{ p: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, flexGrow: 1, flexWrap: 'wrap' }}>
            <TextField
              placeholder="Search papers..."
              variant="outlined"
              size="small"
              value={searchTerm}
              onChange={handleSearchChange}
              sx={{ flexGrow: 1, minWidth: '200px' }}
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
                <MenuItem value="published">Published</MenuItem>
                <MenuItem value="draft">Draft</MenuItem>
                <MenuItem value="review">In Review</MenuItem>
              </Select>
            </FormControl>
          </Box>
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <IconButton 
              color={viewType === 'grid' ? 'primary' : 'default'} 
              onClick={() => handleViewTypeChange('grid')}
            >
              <FilterListIcon />
            </IconButton>
            <IconButton 
              color={viewType === 'list' ? 'primary' : 'default'} 
              onClick={() => handleViewTypeChange('list')}
            >
              <Article />
            </IconButton>
          </Box>
        </Box>
      </Paper>
      
      {/* Grid View */}
      {viewType === 'grid' && (
        <Grid container spacing={3}>
          {filteredPapers.map((paper) => (
            <Grid item xs={12} sm={6} md={4} key={paper.id}>
              <Card 
                sx={{ 
                  cursor: 'pointer', 
                  height: '100%', 
                  display: 'flex', 
                  flexDirection: 'column' 
                }}
                onClick={() => handlePaperClick(paper.id)}
              >
                <CardContent sx={{ flexGrow: 1 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                    <Typography variant="h6" component="div">
                      {paper.title}
                    </Typography>
                    <IconButton 
                      size="small" 
                      onClick={(e) => handleMenuOpen(e, paper.id)}
                    >
                      <MoreVertIcon />
                    </IconButton>
                  </Box>
                  
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    {paper.abstract.substring(0, 120)}...
                  </Typography>
                  
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <ScienceOutlined fontSize="small" sx={{ mr: 1, color: 'primary.main' }} />
                    <Typography variant="body2" color="text.secondary">
                      {paper.experiment.name}
                    </Typography>
                  </Box>
                  
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mb: 2 }}>
                    {paper.tags.map((tag) => (
                      <Chip key={tag} label={tag} size="small" />
                    ))}
                  </Box>
                  
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography variant="caption" color="text.secondary">
                      {paper.date}
                    </Typography>
                    {getStatusChip(paper.status)}
                  </Box>
                </CardContent>
                <Divider />
                <CardActions>
                  <Box sx={{ display: 'flex', width: '100%', justifyContent: 'space-between' }}>
                    <Button size="small" startIcon={<CheckCircle fontSize="small" />}>
                      {paper.citations} Citations
                    </Button>
                    <Button size="small" startIcon={<CloudDownload fontSize="small" />}>
                      {paper.downloads} Downloads
                    </Button>
                  </Box>
                </CardActions>
              </Card>
            </Grid>
          ))}
          
          {filteredPapers.length === 0 && (
            <Grid item xs={12}>
              <Paper sx={{ p: 3, textAlign: 'center' }}>
                <Typography variant="h6" color="text.secondary">
                  No papers found matching your filters.
                </Typography>
              </Paper>
            </Grid>
          )}
        </Grid>
      )}
      
      {/* List View */}
      {viewType === 'list' && (
        <Paper>
          <List>
            {filteredPapers.map((paper) => (
              <React.Fragment key={paper.id}>
                <ListItem 
                  button 
                  onClick={() => handlePaperClick(paper.id)}
                  alignItems="flex-start"
                >
                  <ListItemIcon>
                    <Description color="primary" />
                  </ListItemIcon>
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', pr: 3 }}>
                        <Typography variant="subtitle1" fontWeight={500}>
                          {paper.title}
                        </Typography>
                        {getStatusChip(paper.status)}
                      </Box>
                    }
                    secondary={
                      <>
                        <Typography
                          variant="body2"
                          color="text.primary"
                          sx={{ display: 'block', mb: 1, mt: 0.5 }}
                        >
                          {paper.abstract.substring(0, 150)}...
                        </Typography>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <ScienceOutlined fontSize="small" sx={{ mr: 0.5, color: 'primary.main' }} />
                            <Typography variant="caption" color="text.secondary">
                              {paper.experiment.name}
                            </Typography>
                          </Box>
                          <Typography variant="caption" color="text.secondary">
                            {paper.date}
                          </Typography>
                          <Box>
                            <Typography variant="caption" color="text.secondary" sx={{ mr: 1 }}>
                              {paper.citations} citations
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              {paper.downloads} downloads
                            </Typography>
                          </Box>
                        </Box>
                      </>
                    }
                  />
                  <ListItemSecondaryAction>
                    <IconButton edge="end" onClick={(e) => handleMenuOpen(e, paper.id)}>
                      <MoreVertIcon />
                    </IconButton>
                  </ListItemSecondaryAction>
                </ListItem>
                <Divider variant="inset" component="li" />
              </React.Fragment>
            ))}
            
            {filteredPapers.length === 0 && (
              <ListItem>
                <ListItemText
                  primary={
                    <Typography variant="h6" color="text.secondary" align="center">
                      No papers found matching your filters.
                    </Typography>
                  }
                />
              </ListItem>
            )}
          </List>
        </Paper>
      )}
      
      {/* Paper Actions Menu */}
      <Menu
        anchorEl={menuAnchorEl}
        open={Boolean(menuAnchorEl)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={() => { handleMenuClose(); navigate(`/papers/${selectedPaperId}`); }}>
          <ListItemIcon>
            <Visibility fontSize="small" />
          </ListItemIcon>
          View Paper
        </MenuItem>
        <MenuItem onClick={handleMenuClose}>
          <ListItemIcon>
            <Edit fontSize="small" />
          </ListItemIcon>
          Edit Paper
        </MenuItem>
        <MenuItem onClick={handleMenuClose}>
          <ListItemIcon>
            <CloudDownload fontSize="small" />
          </ListItemIcon>
          Download PDF
        </MenuItem>
        <MenuItem onClick={handleMenuClose}>
          <ListItemIcon>
            <Share fontSize="small" />
          </ListItemIcon>
          Share
        </MenuItem>
        <Divider />
        <MenuItem onClick={handleMenuClose} sx={{ color: 'error.main' }}>
          <ListItemIcon>
            <Delete fontSize="small" sx={{ color: 'error.main' }} />
          </ListItemIcon>
          Delete
        </MenuItem>
      </Menu>
    </Box>
  );
};

export default PapersList; 