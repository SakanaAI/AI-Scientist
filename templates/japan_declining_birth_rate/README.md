# Using the AI Scientist to address Japan's Declining Birth Rate 

This project demonstrates an AI-driven approach to evaluating and optimizing government policies that aim to reverse Japan's declining birth rate. Using The AI Scientist, this project simulates multiple AI-generated policies, applies a neural network model, and identifies the most effective interventions.  


## AI Scientist Generated Paper

[Link to Paper](https://drive.google.com/file/d/1AnR6ZgkgHhiTxMfGM731Heq5QUiDZhNM/view?usp=sharing)  


## Installation

```bash
# Navigate to the project directory
cd templates/japan_declining_birth_rate

# Activate conda environment
conda activate ai_scientist
```

## Running the Baseline Template

1. Train the neural network baseline model:  

```bash
python experiment.py --out_dir run_0
```

2. Generate visualization plots:  

```bash
python plot.py
```


## Running AI Scientist  

1. Initialize and run The AI Scientist:  

```bash
python launch_scientist.py \
    --model "claude-3-5-sonnet-20241022" \
    --experiment japan_declining_birth_rate \
    --num-ideas 1
```

*(Modify `--num-ideas` based on the number of policy variations to explore.)*  


## How It Works  

1. Policy Generation:  
   - AI creates multiple unique policy interventions.  
   - Each policy consists of budget allocation, duration, and an expected impact on birth rates.  

2. Neural Network Modeling:  
   - A simple fully connected neural network learns to predict policy effectiveness.  
   - Inputs: Budget + Duration  
   - Output: Expected birth rate increase  

3. Optimization & Evaluation:  
   - AI iterates through policies to find cost-effective strategies.  
   - Results are saved in `final_info.json`.  


## Credits  

This project builds upon:  

- The AI Scientist System  
  - Repository: [https://github.com/SakanaAI/AI-Scientist](https://github.com/SakanaAI/AI-Scientist)  
