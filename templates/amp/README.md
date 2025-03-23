# AMP Template Setup Guide

## Overview
This guide provides basic details on the AMP algorithm. Additionally, it provides step-by-step instructions for setting up and running the AI Scientist with the AMP template and DeepMimic integration.

## What Is AMP?
Adversarial Motion Priors ```(Peng et al., 2021)``` presents an unsupervised reinforcement learning approach to character animation based on learning from unstructured motion data to cast natural behaviors on simulated characters.

Paper Website available here:
```
https://xbpeng.github.io/projects/AMP/index.html
```

The paper was released with the ```DeepMimic``` library as a framework for training AMP agents. This template for the AI-Scientist allows users to experiment with modifications to the base AMP algorithm within the DeepMimic library.

```DeepMimic``` requires a somewhat complicated build process, so I wrote a bash script ```DeepMimic/auto_setup.sh``` that handles the entire setup process.

The ```experiment.py``` file implements a simple training run on an AMP agent for 3 different motion files:
```
"DeepMimic/data/motions/humanoid3d_walk.txt"
"DeepMimic/data/motions/humanoid3d_jog.txt"
"DeepMimic/data/motions/humanoid3d_run.txt"
```

Anothe popular (and more recent) option for experimenting with AMP is through the [ProtoMotions](https://github.com/NVlabs/ProtoMotions) Library, which uses NVIDIA's IsaacGym as a backbone. For this reason, I decided to go with DeepMimic as a more light-weight alternative that still allows users to test and evaluate experimental conditions on the base AMP algorithm. 

Please follow the section below for specific setup instructions, and please see ```templates/amp/examples/``` for example paper generations. Note, that the Semantic Scholar API was not used for any of these generations, as I am on the waiting list for an API key.

I generated the given example papers on a "fresh-out-the-box" A100 (40 GB SXM4) on Lambda Labs by followings the instructions as indicated in [Step-by-Step Setup Instructions](#setup-instructions).

## Prerequisites
Before beginning the setup process, ensure that you have Miniconda3 installed on your system at ```/home/ubuntu/miniconda3```. If it is not already installed, it will be handled by the ```DeepMimic/auto_setup.sh``` script automatically. This is important because this path is used for building the python wrapper of DeepMimic in
```DeepMimic/DeepMimicCore/Makefile.auto```.

<a id="setup-instructions"></a>
## Step-by-Step Setup Instructions


### Global Environment Setup
```bash
# Create and activate a new conda environment
conda create -n ai_scientist python=3.11
conda activate ai_scientist

# Install LaTeX dependencies
sudo apt-get install texlive-full

# Install required Python packages from AI-Scientist root
pip install -r requirements.txt
```

### DeepMimic Configuration
```bash
# Initialize and update the DeepMimic submodule
git submodule update --init

# Navigate to DeepMimic directory
cd templates/amp/DeepMimic

# Build the Python wrapper for DeepMimicCore
bash auto_setup.sh

# Make sure Conda was exported to PATH if installed through auto_setup.sh
PATH ="/home/ubuntu/miniconda3:$PATH"
echo 'export PATH="/home/ubuntu/miniconda3:$PATH"' >> ~/.bashrc
source ~/.bashrc
```


### Running Experiments
```bash
# Move to the AMP template directory
cd ../

# Execute the experiment
python experiment.py

# Generate visualization plots
python plot.py
```

### Launching AI Scientist
```bash
# Go to AI-Scientist Root Directory
cd ../../

# Ensure you're in the ai_scientist environment
conda activate ai_scientist

# Launch the AI Scientist with specified parameters
python launch_scientist.py --model "gpt-4o-2024-05-13" --experiment amp --num-ideas 2

python launch_scientist.py --model "claude-3-5-sonnet-20241022" --experiment amp --num-ideas 2
```

## Relevant Directory Subset
```
AI-Scientist/
├── launch_scientist.py
├── requirements.txt
├── templates/
│   └── amp/
│       ├── DeepMimic/
│       │   └── auto_setup.sh
│       ├── experiment.py
│       └── plot.py
```