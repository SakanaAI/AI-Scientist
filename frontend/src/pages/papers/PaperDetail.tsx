import React, { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Paper,
  Typography,
  Grid,
  Button,
  Chip,
  Divider,
  Card,
  CardContent,
  IconButton,
  Menu,
  MenuItem,
  ListItemIcon,
  Tab,
  Tabs,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  List,
  ListItem,
  ListItemText,
  ListItemButton
} from '@mui/material';
import {
  Edit,
  Download,
  Share,
  Science,
  Description,
  Assessment,
  ArrowBack,
  CloudDownload,
  Visibility,
  More,
  Image,
  TableChart,
  FormatQuote,
  Favorite,
  FavoriteBorder,
  ChatBubbleOutline,
  CheckCircle,
  Print,
  Delete,
  MoreVert
} from '@mui/icons-material';
import PageHeader from '../../components/common/PageHeader';

// Mock paper data
const paper = {
  id: '1',
  title: 'Understanding Emergent Abilities in Large Language Models',
  abstract: 'This paper explores the phenomenon of emergent abilities in large language models, where capabilities appear suddenly as model scale increases. We provide a framework for identifying and analyzing these emergent abilities, demonstrating that they are more widespread than previously thought. Our analysis shows that these capabilities are not merely statistical artifacts but represent qualitative shifts in model behavior that correspond to meaningful cognitive abilities. We discuss implications for scaling laws, model training, and the development of more capable AI systems.',
  experiment: { id: '1', name: 'Transformer Self-Attention Patterns' },
  status: 'published',
  date: '2023-10-30',
  lastUpdated: '2023-11-05',
  author: 'AI-Scientist',
  tags: ['LLM', 'Scaling', 'Emergent Abilities'],
  citations: 12,
  downloads: 89,
  views: 341,
  content: {
    introduction: 'Recent years have seen rapid advancement in the capabilities of large language models (LLMs). As these models scale in size, they exhibit behaviors that were not present in smaller versions - abilities that seem to emerge suddenly once models reach a certain scale. These "emergent abilities" have attracted significant attention in the research community, but our understanding of the phenomenon remains limited.',
    relatedWork: 'The study of emergent abilities in LLMs builds on several research areas. First, scaling laws research has demonstrated that many aspects of model performance improve predictably as a power law of model size. Second, work on mechanistic interpretability seeks to understand the internal mechanisms that give rise to model capabilities. Third, research on in-context learning has shown that large models can adapt to new tasks without explicit fine-tuning.',
    methodology: 'We conducted a systematic analysis of model performance across a range of tasks at different model scales. Models ranging from 100M to 175B parameters were evaluated on 25 distinct tasks spanning reasoning, translation, common sense, and specialized knowledge. We employed a combination of established benchmarks and novel evaluation protocols designed to isolate specific capabilities.',
    results: 'Our investigation revealed several key findings. First, emergent abilities are more widespread than previously documented, appearing across diverse task categories. Second, the emergence of new abilities follows a pattern where performance remains near random chance until models reach a critical scale, at which point performance rapidly improves before plateauing. Third, these transitions occur at different model scales for different abilities, suggesting a hierarchy of capabilities.',
    discussion: 'The pattern of emergent abilities raises important questions about the nature of learning in deep neural networks. Rather than simply improving gradually with scale, these models appear to undergo phase transitions in their capabilities. This suggests that fundamental cognitive abilities may require a minimum computational capacity, after which they become available to the system.',
    conclusion: 'Our work provides a more comprehensive understanding of emergent abilities in large language models. The framework we develop helps identify, characterize, and predict the emergence of new capabilities as models scale. These insights have implications for the development of more capable AI systems and our understanding of the relationship between scale and intelligence in artificial systems.',
    acknowledgements: 'We thank the entire research team for their contributions to this work. This research was supported by the AI-Scientist platform.'
  },
  references: [
    { id: '1', title: 'Language Models are Few-Shot Learners', authors: 'Brown et al.', year: '2020', venue: 'NeurIPS' },
    { id: '2', title: 'Scaling Laws for Neural Language Models', authors: 'Kaplan et al.', year: '2020', venue: 'arXiv' },
    { id: '3', title: 'Emergent Abilities of Large Language Models', authors: 'Wei et al.', year: '2022', venue: 'TMLR' },
    { id: '4', title: 'Training Compute-Optimal Large Language Models', authors: 'Hoffmann et al.', year: '2022', venue: 'arXiv' }
  ],
  figures: [
    { id: '1', caption: 'Performance on reasoning tasks as a function of model scale', path: '/figures/reasoning_scale.png' },
    { id: '2', caption: 'Critical thresholds for different emergent abilities', path: '/figures/thresholds.png' },
    { id: '3', caption: 'Attention patterns exhibiting phase transitions', path: '/figures/attention_patterns.png' }
  ],
  tables: [
    { id: '1', caption: 'Model performance on benchmark tasks', path: '/tables/benchmarks.csv' },
    { id: '2', caption: 'Scaling coefficients for different ability categories', path: '/tables/scaling_coefficients.csv' }
  ],
  reviews: [
    { id: '1', reviewer: 'Reviewer 1', content: 'The paper presents a comprehensive analysis of emergent abilities in LLMs. The framework developed is valuable for understanding scaling behaviors.', rating: 9 },
    { id: '2', reviewer: 'Reviewer 2', content: 'This work provides important insights into the nature of capabilities that emerge with scale. However, more attention to the limitations of the evaluation methodology would strengthen the paper.', rating: 8 }
  ]
};

// Interface for tab panel props
interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`paper-tabpanel-${index}`}
      aria-labelledby={`paper-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  );
}

const PaperDetail: React.FC = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  
  // State
  const [tabValue, setTabValue] = useState(0);
  const [menuAnchorEl, setMenuAnchorEl] = useState<null | HTMLElement>(null);
  const [isFavorite, setIsFavorite] = useState(false);
  
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };
  
  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setMenuAnchorEl(event.currentTarget);
  };
  
  const handleMenuClose = () => {
    setMenuAnchorEl(null);
  };
  
  const toggleFavorite = () => {
    setIsFavorite(!isFavorite);
  };
  
  const getStatusChip = (status: string) => {
    switch (status) {
      case 'published':
        return <Chip icon={<CheckCircle fontSize="small" />} label="Published" color="success" />;
      case 'draft':
        return <Chip icon={<Edit fontSize="small" />} label="Draft" color="default" />;
      case 'review':
        return <Chip icon={<Visibility fontSize="small" />} label="In Review" color="primary" />;
      default:
        return <Chip label={status} color="default" />;
    }
  };
  
  // Breadcrumbs for the page header
  const breadcrumbs = [
    { label: 'Papers', href: '/papers' },
    { label: paper.title }
  ];
  
  return (
    <Box>
      <PageHeader
        title={paper.title}
        subtitle={`Published: ${paper.date} • Last updated: ${paper.lastUpdated}`}
        breadcrumbs={breadcrumbs}
        action={{
          label: "Back to Papers",
          icon: <ArrowBack />,
          onClick: () => navigate('/papers')
        }}
      />
      
      {/* Paper Header */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={8}>
            <Box sx={{ mb: 2 }}>
              <Typography variant="h4" gutterBottom>
                {paper.title}
              </Typography>
              <Typography variant="subtitle1" gutterBottom>
                <strong>Author:</strong> {paper.author}
              </Typography>
            </Box>
            
            <Box sx={{ mb: 2 }}>
              <Typography variant="h6" gutterBottom>
                Abstract
              </Typography>
              <Typography variant="body1" paragraph>
                {paper.abstract}
              </Typography>
            </Box>
            
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
              {paper.tags.map((tag) => (
                <Chip key={tag} label={tag} color="primary" />
              ))}
            </Box>
            
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <Science color="primary" sx={{ mr: 1 }} />
              <Typography variant="body2">
                Based on experiment: <strong>{paper.experiment.name}</strong>
              </Typography>
            </Box>
          </Grid>
          
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6">Paper Stats</Typography>
                  {getStatusChip(paper.status)}
                </Box>
                
                <Grid container spacing={2}>
                  <Grid item xs={4}>
                    <Typography variant="subtitle2" color="text.secondary">
                      Citations
                    </Typography>
                    <Typography variant="h5">
                      {paper.citations}
                    </Typography>
                  </Grid>
                  <Grid item xs={4}>
                    <Typography variant="subtitle2" color="text.secondary">
                      Downloads
                    </Typography>
                    <Typography variant="h5">
                      {paper.downloads}
                    </Typography>
                  </Grid>
                  <Grid item xs={4}>
                    <Typography variant="subtitle2" color="text.secondary">
                      Views
                    </Typography>
                    <Typography variant="h5">
                      {paper.views}
                    </Typography>
                  </Grid>
                </Grid>
                
                <Divider sx={{ my: 2 }} />
                
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Button 
                    variant="contained" 
                    startIcon={<CloudDownload />}
                    fullWidth
                    sx={{ mr: 1 }}
                  >
                    Download PDF
                  </Button>
                  <IconButton 
                    onClick={toggleFavorite}
                    color={isFavorite ? 'primary' : 'default'}
                  >
                    {isFavorite ? <Favorite /> : <FavoriteBorder />}
                  </IconButton>
                  <IconButton onClick={handleMenuOpen}>
                    <MoreVert />
                  </IconButton>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Paper>
      
      {/* Paper Content Tabs */}
      <Box sx={{ mb: 3 }}>
        <Paper sx={{ mb: 0.5 }}>
          <Tabs 
            value={tabValue} 
            onChange={handleTabChange}
            variant="scrollable"
            scrollButtons="auto"
            aria-label="paper content tabs"
          >
            <Tab label="Full Paper" icon={<Description />} />
            <Tab label="Figures & Tables" icon={<Image />} />
            <Tab label="References" icon={<FormatQuote />} />
            <Tab label="Reviews" icon={<Assessment />} />
          </Tabs>
        </Paper>
        
        {/* Full Paper Content */}
        <TabPanel value={tabValue} index={0}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h5" gutterBottom>
              Introduction
            </Typography>
            <Typography variant="body1" paragraph>
              {paper.content.introduction}
            </Typography>
            
            <Typography variant="h5" gutterBottom>
              Related Work
            </Typography>
            <Typography variant="body1" paragraph>
              {paper.content.relatedWork}
            </Typography>
            
            <Typography variant="h5" gutterBottom>
              Methodology
            </Typography>
            <Typography variant="body1" paragraph>
              {paper.content.methodology}
            </Typography>
            
            <Typography variant="h5" gutterBottom>
              Results
            </Typography>
            <Typography variant="body1" paragraph>
              {paper.content.results}
            </Typography>
            
            <Typography variant="h5" gutterBottom>
              Discussion
            </Typography>
            <Typography variant="body1" paragraph>
              {paper.content.discussion}
            </Typography>
            
            <Typography variant="h5" gutterBottom>
              Conclusion
            </Typography>
            <Typography variant="body1" paragraph>
              {paper.content.conclusion}
            </Typography>
            
            <Divider sx={{ my: 3 }} />
            
            <Typography variant="h6" gutterBottom>
              Acknowledgements
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              {paper.content.acknowledgements}
            </Typography>
          </Paper>
        </TabPanel>
        
        {/* Figures & Tables */}
        <TabPanel value={tabValue} index={1}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h5" gutterBottom>
                  Figures
                </Typography>
                <Divider sx={{ mb: 2 }} />
                
                <Grid container spacing={3}>
                  {paper.figures.map((figure) => (
                    <Grid item xs={12} md={4} key={figure.id}>
                      <Card>
                        <Box sx={{ height: 200, bgcolor: 'grey.200', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                          <Image fontSize="large" color="disabled" />
                        </Box>
                        <CardContent>
                          <Typography variant="subtitle1" gutterBottom>
                            Figure {figure.id}: {figure.caption}
                          </Typography>
                          <Button size="small" startIcon={<Visibility />}>
                            View Full Size
                          </Button>
                        </CardContent>
                      </Card>
                    </Grid>
                  ))}
                </Grid>
              </Paper>
            </Grid>
            
            <Grid item xs={12}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h5" gutterBottom>
                  Tables
                </Typography>
                <Divider sx={{ mb: 2 }} />
                
                <Grid container spacing={3}>
                  {paper.tables.map((table) => (
                    <Grid item xs={12} key={table.id}>
                      <Card>
                        <CardContent>
                          <Typography variant="subtitle1" gutterBottom>
                            Table {table.id}: {table.caption}
                          </Typography>
                          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', p: 2, bgcolor: 'grey.50' }}>
                            <TableChart fontSize="large" color="primary" />
                          </Box>
                          <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
                            <Button size="small" startIcon={<Visibility />}>
                              View Table
                            </Button>
                            <Button size="small" startIcon={<CloudDownload />} sx={{ ml: 1 }}>
                              Download CSV
                            </Button>
                          </Box>
                        </CardContent>
                      </Card>
                    </Grid>
                  ))}
                </Grid>
              </Paper>
            </Grid>
          </Grid>
        </TabPanel>
        
        {/* References */}
        <TabPanel value={tabValue} index={2}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h5" gutterBottom>
              References
            </Typography>
            <Divider sx={{ mb: 2 }} />
            
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>#</TableCell>
                    <TableCell>Title</TableCell>
                    <TableCell>Authors</TableCell>
                    <TableCell>Year</TableCell>
                    <TableCell>Venue</TableCell>
                    <TableCell align="right">Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {paper.references.map((reference, index) => (
                    <TableRow key={reference.id}>
                      <TableCell>{index + 1}</TableCell>
                      <TableCell>{reference.title}</TableCell>
                      <TableCell>{reference.authors}</TableCell>
                      <TableCell>{reference.year}</TableCell>
                      <TableCell>{reference.venue}</TableCell>
                      <TableCell align="right">
                        <Button size="small">
                          View
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </TabPanel>
        
        {/* Reviews */}
        <TabPanel value={tabValue} index={3}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h5" gutterBottom>
              Reviews
            </Typography>
            <Divider sx={{ mb: 2 }} />
            
            {paper.reviews.map((review) => (
              <Card key={review.id} sx={{ mb: 2 }}>
                <CardContent>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                    <Typography variant="h6">
                      {review.reviewer}
                    </Typography>
                    <Chip 
                      label={`Rating: ${review.rating}/10`} 
                      color={review.rating >= 8 ? 'success' : review.rating >= 6 ? 'primary' : 'default'} 
                    />
                  </Box>
                  <Typography variant="body1">
                    {review.content}
                  </Typography>
                </CardContent>
              </Card>
            ))}
            
            <Box sx={{ mt: 3, textAlign: 'center' }}>
              <Button variant="outlined" startIcon={<Assessment />}>
                Request Additional Review
              </Button>
            </Box>
          </Paper>
        </TabPanel>
      </Box>
      
      {/* Paper Actions Menu */}
      <Menu
        anchorEl={menuAnchorEl}
        open={Boolean(menuAnchorEl)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={handleMenuClose}>
          <ListItemIcon>
            <Edit fontSize="small" />
          </ListItemIcon>
          Edit Paper
        </MenuItem>
        <MenuItem onClick={handleMenuClose}>
          <ListItemIcon>
            <Print fontSize="small" />
          </ListItemIcon>
          Print
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
          Delete Paper
        </MenuItem>
      </Menu>
    </Box>
  );
};

export default PaperDetail; 