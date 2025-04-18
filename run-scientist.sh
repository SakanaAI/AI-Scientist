#!/bin/bash

# Colors for prettier output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default values
OPERATION="run"
EXPERIMENT="nanoGPT"
MODEL="claude-3-5-sonnet-20241022"
NUM_IDEAS=2
VERBOSE=false

# Help message
function show_help {
  echo -e "${BLUE}AI-Scientist Runner${NC}"
  echo
  echo "Usage: ./run-scientist.sh [OPTIONS]"
  echo
  echo "Operations:"
  echo "  --operation OPERATION  Operation to perform (default: run)"
  echo "                         Available operations: run, frontend, process-data, visualize"
  echo
  echo "Options:"
  echo "  --experiment TYPE      Experiment type to run (default: nanoGPT)"
  echo "                         Available types: nanoGPT, 2d_diffusion, grokking, custom"
  echo "  --model MODEL          LLM model to use (default: claude-3-5-sonnet-20241022)"
  echo "  --num-ideas NUM        Number of ideas to generate (default: 2)"
  echo "  --verbose              Enable verbose output"
  echo "  --help                 Display this help message"
  echo
  echo "Examples:"
  echo "  ./run-scientist.sh                                   # Run with default settings"
  echo "  ./run-scientist.sh --experiment 2d_diffusion         # Run 2D diffusion experiment"
  echo "  ./run-scientist.sh --operation frontend              # Start the frontend"
  echo "  ./run-scientist.sh --operation process-data          # Process experimental data"
}

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --operation) OPERATION="$2"; shift ;;
    --experiment) EXPERIMENT="$2"; shift ;;
    --model) MODEL="$2"; shift ;;
    --num-ideas) NUM_IDEAS="$2"; shift ;;
    --verbose) VERBOSE=true ;;
    --help) show_help; exit 0 ;;
    *) echo "Unknown parameter: $1"; show_help; exit 1 ;;
  esac
  shift
done

# Create output directories if they don't exist
mkdir -p output_science/data/raw
mkdir -p output_science/data/processed
mkdir -p output_science/experiments/$EXPERIMENT
mkdir -p output_science/papers/drafts
mkdir -p output_science/papers/final
mkdir -p output_science/logs
mkdir -p output_science/results

# Validate experiment type
valid_experiments=("nanoGPT" "2d_diffusion" "grokking" "custom")
if [[ ! " ${valid_experiments[@]} " =~ " ${EXPERIMENT} " ]]; then
  echo -e "${RED}Error: Invalid experiment type '${EXPERIMENT}'${NC}"
  echo -e "Valid experiment types: ${valid_experiments[*]}"
  exit 1
fi

# Run the appropriate operation
case $OPERATION in
  "run")
    echo -e "${GREEN}Starting AI-Scientist with the following settings:${NC}"
    echo -e "  Experiment: ${BLUE}$EXPERIMENT${NC}"
    echo -e "  Model: ${BLUE}$MODEL${NC}"
    echo -e "  Number of ideas: ${BLUE}$NUM_IDEAS${NC}"
    
    if [ "$VERBOSE" = true ]; then
      python launch_scientist.py --model "$MODEL" --experiment "$EXPERIMENT" --num-ideas "$NUM_IDEAS" --verbose
    else
      python launch_scientist.py --model "$MODEL" --experiment "$EXPERIMENT" --num-ideas "$NUM_IDEAS"
    fi
    ;;
    
  "frontend")
    echo -e "${GREEN}Starting AI-Scientist Frontend...${NC}"
    cd frontend && ./run-frontend.sh
    ;;
    
  "process-data")
    echo -e "${GREEN}Processing data for ${BLUE}$EXPERIMENT${NC} experiment...${NC}"
    python scripts/data_processing/process_data.py --experiment "$EXPERIMENT"
    ;;
    
  "visualize")
    echo -e "${GREEN}Visualizing results for ${BLUE}$EXPERIMENT${NC} experiment...${NC}"
    python scripts/visualization/visualize_results.py --experiment "$EXPERIMENT"
    ;;
    
  *)
    echo -e "${RED}Error: Unknown operation '${OPERATION}'${NC}"
    show_help
    exit 1
    ;;
esac 