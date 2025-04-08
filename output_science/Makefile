# AI Scientist Makefile
# Provides shortcuts for common operations

PYTHON=python3
BASE_DIR=$(shell pwd)
SCRIPTS_DIR=$(BASE_DIR)/scripts
OUTPUT_DIR=$(BASE_DIR)

.PHONY: help setup data-nanogpt data-2d-diffusion data-grokking run-nanogpt run-2d-diffusion run-grokking \
		visualize-nanogpt visualize-2d-diffusion visualize-grokking review all clean

help:
	@echo "AI Scientist Project Makefile"
	@echo ""
	@echo "Available commands:"
	@echo "  make setup              - Create all necessary directories and config files"
	@echo "  make data-nanogpt       - Process data for nanoGPT experiments"
	@echo "  make data-2d-diffusion  - Process data for 2D diffusion experiments"
	@echo "  make data-grokking      - Process data for grokking experiments"
	@echo "  make run-nanogpt        - Run nanoGPT experiments"
	@echo "  make run-2d-diffusion   - Run 2D diffusion experiments"
	@echo "  make run-grokking       - Run grokking experiments"
	@echo "  make visualize-nanogpt  - Visualize nanoGPT experiment results"
	@echo "  make visualize-2d-diffusion - Visualize 2D diffusion experiment results"
	@echo "  make visualize-grokking - Visualize grokking experiment results"
	@echo "  make review             - Generate reviews for a paper"
	@echo "  make clean              - Clean up temporary files"
	@echo ""
	@echo "Variables:"
	@echo "  MODEL       - LLM model to use (default: claude-3-5-sonnet-20241022)"
	@echo "  NUM_IDEAS   - Number of ideas to generate (default: 5)"
	@echo "  IDEA_ID     - Specific idea ID to process (optional)"
	@echo ""

setup:
	@echo "Setting up project directories..."
	mkdir -p $(OUTPUT_DIR)/{data/{raw,processed},experiments/{nanoGPT,2d_diffusion,grokking,custom},papers/{drafts,final,assets},models/{checkpoints,final},logs/{idea_generation,experiments,paper_writing},reviews,workflows/{idea_generation,experimentation,paper_writing},scripts/{data_processing,visualization,evaluation},results/{metrics,comparisons,novelty},config/{models,templates,experiments}}
	@echo "Setup complete."

data-nanogpt:
	@echo "Processing nanoGPT data..."
	$(PYTHON) $(SCRIPTS_DIR)/data_processing/process_datasets.py --experiment nanoGPT

data-2d-diffusion:
	@echo "Processing 2D diffusion data..."
	$(PYTHON) $(SCRIPTS_DIR)/data_processing/process_datasets.py --experiment 2d_diffusion

data-grokking:
	@echo "Processing grokking data..."
	$(PYTHON) $(SCRIPTS_DIR)/data_processing/process_datasets.py --experiment grokking

run-nanogpt:
	@echo "Running nanoGPT experiments..."
	$(PYTHON) $(SCRIPTS_DIR)/launch_scientist_custom.py --experiment nanoGPT --model $(MODEL) --num-ideas $(NUM_IDEAS)

run-nanogpt-lite:
	@echo "Running nanoGPT lite experiments..."
	$(PYTHON) $(SCRIPTS_DIR)/launch_scientist_custom.py --experiment nanoGPT_lite --model $(MODEL) --num-ideas $(NUM_IDEAS)

run-2d-diffusion:
	@echo "Running 2D diffusion experiments..."
	$(PYTHON) $(SCRIPTS_DIR)/launch_scientist_custom.py --experiment 2d_diffusion --model $(MODEL) --num-ideas $(NUM_IDEAS)

run-grokking:
	@echo "Running grokking experiments..."
	$(PYTHON) $(SCRIPTS_DIR)/launch_scientist_custom.py --experiment grokking --model $(MODEL) --num-ideas $(NUM_IDEAS)

visualize-nanogpt:
	@echo "Visualizing nanoGPT results..."
	$(PYTHON) $(SCRIPTS_DIR)/visualization/plot_results.py --experiment nanoGPT $(if $(IDEA_ID),--idea-id $(IDEA_ID),)

visualize-2d-diffusion:
	@echo "Visualizing 2D diffusion results..."
	$(PYTHON) $(SCRIPTS_DIR)/visualization/plot_results.py --experiment 2d_diffusion $(if $(IDEA_ID),--idea-id $(IDEA_ID),)

visualize-grokking:
	@echo "Visualizing grokking results..."
	$(PYTHON) $(SCRIPTS_DIR)/visualization/plot_results.py --experiment grokking $(if $(IDEA_ID),--idea-id $(IDEA_ID),)

review:
	@echo "Generating paper review..."
	@if [ -z "$(IDEA_ID)" ]; then \
		echo "Error: IDEA_ID not specified. Use 'make review IDEA_ID=your_idea_id'"; \
		exit 1; \
	fi
	$(PYTHON) $(SCRIPTS_DIR)/launch_scientist_custom.py --experiment $(EXPERIMENT) --model $(MODEL) --skip-idea-generation --skip-experiment --skip-writeup --review --idea-id $(IDEA_ID)

# Default values
MODEL ?= claude-3-5-sonnet-20241022
NUM_IDEAS ?= 5
EXPERIMENT ?= nanoGPT

# Clean temporary files
clean:
	@echo "Cleaning up temporary files..."
	find $(OUTPUT_DIR) -name "*.pyc" -delete
	find $(OUTPUT_DIR) -name "__pycache__" -delete
	find $(OUTPUT_DIR) -name "*.log" -delete
	@echo "Clean complete." 