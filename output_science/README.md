# AI-Scientist Project Structure

This document outlines the full structure of the AI-Scientist project.

## Project Overview

AI-Scientist is a comprehensive system for automated scientific discovery, enabling Foundation Models like Large Language Models (LLMs) to perform research independently. The system generates ideas, conducts experiments, and produces scientific papers in various domains.

## Directory Structure

```
output_science/                  # Main output directory for AI-Scientist
├── data/                        # All datasets and processed data
│   ├── raw/                     # Raw unprocessed data
│   └── processed/               # Processed data ready for experiments
├── experiments/                 # Experiment outputs organized by template
│   ├── nanoGPT/                 # NanoGPT experiment results
│   ├── 2d_diffusion/            # 2D Diffusion experiment results
│   ├── grokking/                # Grokking experiment results
│   └── custom/                  # Results from custom templates
├── papers/                      # Generated scientific papers
│   ├── drafts/                  # Paper drafts in progress
│   ├── final/                   # Finalized papers
│   └── assets/                  # Figures, tables, and other paper assets
├── models/                      # Trained models and checkpoints
│   ├── checkpoints/             # Model checkpoints during training
│   └── final/                   # Final trained models
├── logs/                        # Log files from runs
│   ├── idea_generation/         # Logs from idea generation phase
│   ├── experiments/             # Logs from experiment execution
│   └── paper_writing/           # Logs from paper writing process
├── reviews/                     # LLM-generated reviews of papers
├── workflows/                   # Workflow definitions and configurations
│   ├── idea_generation/         # Idea generation workflow configs
│   ├── experimentation/         # Experimentation workflow configs
│   └── paper_writing/           # Paper writing workflow configs
├── scripts/                     # Utility scripts for various tasks
│   ├── data_processing/         # Scripts for data preparation
│   ├── visualization/           # Scripts for plotting and visualization
│   └── evaluation/              # Scripts for evaluating results
├── results/                     # Final consolidated results
│   ├── metrics/                 # Performance metrics across experiments
│   ├── comparisons/             # Comparative analyses
│   └── novelty/                 # Novelty assessments of ideas
└── config/                      # Configuration files
    ├── models/                  # LLM model configurations
    ├── templates/               # Template configurations
    └── experiments/             # Experiment configurations
```

Additionally, at the project root level:

```
frontend/                     # Web interface for the AI-Scientist (at project root)
├── public/                   # Static files
├── src/                      # Source code
│   ├── components/           # UI components
│   ├── pages/                # Page components
│   ├── services/             # API services
│   └── context/              # React context providers
├── dist/                     # Production build output
└── package.json              # NPM dependencies and scripts
```

## Key Workflow Components

1. **Idea Generation**: AI models generate novel research ideas in specific domains.
2. **Novelty Check**: Ideas are checked for novelty against existing literature.
3. **Experimentation**: The system conducts experiments to test the generated ideas.
4. **Analysis**: Results are analyzed to validate or refute hypotheses.
5. **Paper Writing**: A complete scientific paper is generated based on the research.
6. **Review**: The system can generate reviews of the papers and suggest improvements.

## Web Interface

The AI-Scientist system includes a modern web interface built with React and Material UI that provides:

1. **User Management**: Authentication and profile management
2. **Project Dashboard**: Overview of research activities
3. **Experiment Management**: Interface for creating and monitoring experiments
4. **Results Visualization**: Visual exploration of experimental results
5. **Paper Generation**: Interface for creating and editing scientific papers

To run the web interface:

```bash
cd frontend
npm install     # Install dependencies
npm start       # Start development server
# OR
npm run build   # Build for production
```

## Usage Instructions

The primary entry point is `launch_scientist.py` in the root directory, which coordinates the entire research process. See the main README for detailed usage instructions. 