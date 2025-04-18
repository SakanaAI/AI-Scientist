#!/bin/bash

# Colors for prettier output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting AI-Scientist Frontend...${NC}"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo -e "${YELLOW}Node.js is not installed. Please install Node.js to run the frontend.${NC}"
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo -e "${YELLOW}npm is not installed. Please install npm to run the frontend.${NC}"
    exit 1
fi

# Check if dependencies are installed
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    npm install
fi

# Start the development server
echo -e "${GREEN}Starting development server...${NC}"
npm start 