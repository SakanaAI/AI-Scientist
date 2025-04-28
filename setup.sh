#!/bin/bash

# AI-Scientist Setup Script
# This script sets up both the backend and frontend components

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}AI-Scientist Setup${NC}"
echo "======================"
echo "This script will set up the AI-Scientist environment."
echo

# Check if conda is installed
if ! command -v conda &>/dev/null; then
    echo -e "${RED}Conda is not installed. Please install Conda first.${NC}"
    echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create conda environment
echo -e "${BLUE}Creating conda environment...${NC}"
conda create -n ai_scientist python=3.11 -y

# Function to check command status
check_status() {
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: $1${NC}"
        exit 1
    fi
}

# Activate environment
echo -e "${BLUE}Activating conda environment...${NC}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ai_scientist
check_status "Failed to activate conda environment"

# Install python dependencies
echo -e "${BLUE}Installing Python dependencies...${NC}"
pip install -r requirements.txt
check_status "Failed to install Python dependencies"

# Ensure output_science directory exists
if [ ! -d "output_science" ]; then
    echo -e "${BLUE}Creating output_science directory structure...${NC}"
    mkdir -p output_science/{data/{raw,processed},experiments/{nanoGPT,2d_diffusion,grokking,custom},papers/{drafts,final,assets},models/{checkpoints,final},logs/{idea_generation,experiments,paper_writing},reviews,workflows/{idea_generation,experimentation,paper_writing},scripts/{data_processing,visualization,evaluation},results/{metrics,comparisons,novelty},config/{models,templates,experiments}}
    check_status "Failed to create output_science directory structure"
fi

# Check if Node.js is installed
if command -v node &>/dev/null; then
    NODE_VERSION=$(node -v)
    echo -e "${BLUE}Node.js is installed: ${NODE_VERSION}${NC}"
    
    # Check if npm is installed
    if command -v npm &>/dev/null; then
        echo -e "${BLUE}Setting up frontend dependencies...${NC}"
        cd frontend
        npm install
        check_status "Failed to install frontend dependencies"
        cd ..
    else
        echo -e "${YELLOW}npm is not installed. Skipping frontend setup.${NC}"
    fi
else
    echo -e "${YELLOW}Node.js is not installed. Skipping frontend setup.${NC}"
    echo "To run the frontend, install Node.js from https://nodejs.org/"
fi

# Make scripts executable
echo -e "${BLUE}Making scripts executable...${NC}"
chmod +x run-scientist.sh
chmod +x frontend/run-frontend.sh

echo
echo -e "${GREEN}Setup complete!${NC}"
echo
echo "To activate the environment and run AI-Scientist:"
echo -e "  ${YELLOW}conda activate ai_scientist${NC}"
echo -e "  ${YELLOW}./run-scientist.sh${NC}"
echo
echo "To run the frontend:"
echo -e "  ${YELLOW}./run-scientist.sh --operation frontend${NC}"
echo 