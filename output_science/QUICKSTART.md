# AI-Scientist Quick Start Guide

This guide will help you get started with using the AI-Scientist project with the new structured output directory.

## Prerequisites

- Python 3.11+
- CUDA-compatible GPU (for experiments)
- Required Python packages (see main README)
- API keys for LLM access (OpenAI, Anthropic, etc.)

## Setup

1. Clone the repository and install dependencies:

```bash
git clone https://github.com/SakanaAI/AI-Scientist.git
cd AI-Scientist
conda create -n ai_scientist python=3.11
conda activate ai_scientist
pip install -r requirements.txt
```

2. Navigate to the output_science directory and set up the project structure:

```bash
cd output_science
make setup
```

3. Set environment variables for API keys:

```bash
export OPENAI_API_KEY="your_openai_key"
# OR
export ANTHROPIC_API_KEY="your_anthropic_key"
# Optional - for literature search
export S2_API_KEY="your_semantic_scholar_key"
```

## Data Preparation

Process data for the experiment you want to run:

```bash
# For nanoGPT experiments
make data-nanogpt

# For 2D diffusion experiments
make data-2d-diffusion

# For grokking experiments
make data-grokking
```

## Running Experiments

Run experiments with the desired model:

```bash
# For nanoGPT with Claude
make run-nanogpt MODEL=claude-3-5-sonnet-20241022 NUM_IDEAS=3

# For 2D diffusion with GPT-4o
make run-2d-diffusion MODEL=gpt-4o-2024-05-13 NUM_IDEAS=3

# For grokking with Claude
make run-grokking MODEL=claude-3-5-sonnet-20241022 NUM_IDEAS=3
```

You can customize the model and number of ideas using the MODEL and NUM_IDEAS variables.

## Visualizing Results

Visualize experiment results:

```bash
# Visualize all nanoGPT experiments
make visualize-nanogpt

# Visualize a specific 2D diffusion experiment
make visualize-2d-diffusion IDEA_ID=your_idea_id

# Visualize all grokking experiments
make visualize-grokking
```

## Getting Reviews

Generate a review for a specific paper:

```bash
make review IDEA_ID=your_idea_id EXPERIMENT=nanoGPT MODEL=claude-3-5-sonnet-20241022
```

## Custom Script Usage

You can directly use the custom scripts for more control:

```bash
# Generate ideas and run experiments
python scripts/launch_scientist_custom.py --experiment nanoGPT --model claude-3-5-sonnet-20241022 --num-ideas 3

# Only generate ideas (skip experiments and writeup)
python scripts/launch_scientist_custom.py --experiment nanoGPT --model claude-3-5-sonnet-20241022 --num-ideas 3 --skip-experiment --skip-writeup

# Process data
python scripts/data_processing/process_datasets.py --experiment nanoGPT

# Visualize results
python scripts/visualization/plot_results.py --experiment nanoGPT --idea-id your_idea_id --format pdf
```

## Directory Structure

All outputs are organized in the `output_science` directory:

- `experiments/`: Contains experiment results by template and idea
- `papers/`: Contains generated papers (drafts and final versions)
- `data/`: Contains raw and processed data
- `logs/`: Contains logs from runs
- `reviews/`: Contains LLM-generated paper reviews

For a complete overview of the directory structure, see the README.md file in the output_science directory.

## Troubleshooting

- **API key issues**: Ensure your API keys are correctly set in environment variables
- **GPU memory errors**: Reduce batch sizes or model sizes in experiment configs
- **Missing dependencies**: Check that all required packages are installed
- **File permissions**: Ensure script files are executable (`chmod +x scripts/*.py`)

For more detailed information, refer to the main project README.md file. 