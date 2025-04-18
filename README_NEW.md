# AI-Scientist

<div align="center">
  <a href="https://github.com/SakanaAI/AI-Scientist/blob/main/docs/logo_2.png">
    <img src="docs/logo_2.png" width="215" alt="AI-Scientist Logo"/>
  </a>
  <h1>The AI Scientist: Towards Fully Automated</h1>
  <h1>Open-Ended Scientific Discovery 🧑‍🔬</h1>
</div>

<p align="center">
  📚 <a href="https://arxiv.org/abs/2408.06292">[Paper]</a> |
  📝 <a href="https://sakana.ai/ai-scientist/">[Blog Post]</a> |
  📂 <a href="https://drive.google.com/drive/folders/1G7A0wTqfXVa-cpexjk0oaXakaSJwffEt">[Drive Folder]</a>
</p>

One of the grand challenges of artificial intelligence is developing agents capable of conducting scientific research and discovering new knowledge. While frontier models have already been used to aid human scientists, they still require extensive manual supervision or are heavily constrained to specific tasks.

We introduce **The AI Scientist**, the first comprehensive system for fully automatic scientific discovery, enabling Foundation Models such as Large Language Models (LLMs) to perform research independently.

## Project Structure

The AI-Scientist project is organized as follows:

```
AI-Scientist/
├── ai_scientist/            # Core AI-Scientist framework
│   ├── generate_ideas.py    # Idea generation module
│   ├── perform_experiments.py # Experiment execution
│   ├── perform_review.py    # Paper review generation
│   ├── perform_writeup.py   # Paper writing module
│   └── ...
├── frontend/                # Web interface (React/TypeScript)
│   ├── public/              # Static files
│   ├── src/                 # Source code
│   │   ├── components/      # UI components
│   │   ├── pages/           # Page components
│   │   ├── services/        # API services
│   │   └── ...
│   ├── package.json         # Frontend dependencies
│   └── run-frontend.sh      # Script to run the frontend
├── templates/               # Experiment templates
│   ├── nanoGPT/             # NanoGPT template
│   ├── 2d_diffusion/        # 2D diffusion template
│   ├── grokking/            # Grokking template
│   └── ...
├── output_science/          # Organized output directory
│   ├── data/                # All datasets and processed data
│   ├── experiments/         # Experiment outputs by template
│   ├── papers/              # Generated papers
│   ├── results/             # Result analysis
│   └── ...
├── launch_scientist.py      # Original launch script
├── run-scientist.sh         # New script for using the custom structure
└── requirements.txt         # Python dependencies
```

## New Features

This repository includes several enhancements to the original AI-Scientist:

1. **Web Interface**: A modern React/TypeScript frontend for managing and visualizing research
2. **Organized Output Structure**: All outputs now stored in a structured `output_science` directory
3. **Improved Scripts**: Shell scripts for common operations and better usability

## Installation

### Backend

```bash
# Create and activate conda environment
conda create -n ai_scientist python=3.11
conda activate ai_scientist

# Install dependencies
pip install -r requirements.txt
```

### Frontend

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

## Running AI-Scientist

You can use the provided shell scripts to interact with the AI-Scientist:

### Using the Web Interface

```bash
# Run the frontend development server
./run-scientist.sh --operation frontend

# Or directly from the frontend directory
cd frontend
./run-frontend.sh
```

### Running Experiments

```bash
# Run with default settings (nanoGPT, Claude model, 5 ideas)
./run-scientist.sh

# Specify experiment type, model, and number of ideas
./run-scientist.sh --experiment 2d_diffusion --model "gpt-4o-2024-05-13" --num-ideas 3

# Process data for an experiment
./run-scientist.sh --operation process-data --experiment nanoGPT

# Visualize experiment results
./run-scientist.sh --operation visualize --experiment grokking
```

### Traditional Usage

You can also use the original launch script for backward compatibility:

```bash
python launch_scientist.py --model "claude-3-5-sonnet-20241022" --experiment nanoGPT --num-ideas 2
```

## Output Directory Structure

All outputs are organized in the `output_science` directory:

- `experiments/`: Contains experiment results by template and idea
- `papers/`: Contains generated papers (drafts and final versions)
- `data/`: Contains raw and processed data
- `logs/`: Contains logs from runs
- `reviews/`: Contains LLM-generated paper reviews

## Frontend Features

The web interface provides:

- **User Authentication**: Secure login and user management
- **Dashboard**: Overview of research activities
- **Project Management**: Create and manage research projects
- **Experiment Orchestration**: Design, run, and monitor experiments
- **Results Visualization**: Interactive charts and data exploration
- **Paper Generation**: Generate and edit scientific papers

## Contributing

We welcome contributions to the AI-Scientist project. Please feel free to submit issues and pull requests.

## License

This project is licensed under the terms in the LICENSE file.

## Citation

If you use the AI-Scientist in your research, please cite our paper:

```bibtex
@article{aiscientist2024,
  title={The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery},
  author={...},
  journal={arXiv preprint arXiv:2408.06292},
  year={2024}
}
```

## Disclaimer

This codebase will execute LLM-written code. There are various risks and challenges associated with this autonomy, including the use of potentially dangerous packages, web access, and potential spawning of processes. Use at your own discretion. Please make sure to containerize and restrict web access appropriately. 