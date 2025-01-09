# Japan Earthquake Prediction Using AI Scientist

This repository demonstrates an advanced approach to earthquake prediction using The AI Scientist. The project focuses on predicting seismic events in Japan.

## AI Scientist Generated Paper

[Link to Paper](https://drive.google.com/file/d/1fHFccphl8xtDEC_nRLfYVhJ3B28kfquu/view?usp=sharing)

## Installation

```bash
# Navigate to the project directory
cd templates/earthquake-prediction

# Activate conda environment
conda activate ai_scientist

# Install required packages
pip install scikit-learn
```

## Dataset Preparation

The dataset includes seismic readings from Japan (200×250 grid cells, 64-day observation windows).

1. Download and preprocess the data:

```bash
python ./data/prepare.py
```

## Running the Baseline Template

1. Train the LSTM baseline model:

```bash
python experiment.py --out_dir run_0
```

2. Generate visualization plots:

```bash
python plot.py
```

## Running AI Scientist

1. Initialize and run the AI Scientist:

```bash
python launch_scientist.py \
    --model "claude-3-5-sonnet-20241022" \
    --experiment earthquake-prediction \
    --num-ideas 1
```

## Credits

This project builds upon the following works:

- **The AI Scientist System**
    - Repository: https://github.com/SakanaAI/AI-Scientist

- **Baseline LSTM Implementation**
    - Repository: https://github.com/romakail/Earthquake_prediction_DNN
