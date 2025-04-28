# Hardware Requirements

This document outlines the hardware requirements for running AI-Scientist effectively.

## Minimum Requirements

These are the minimum specifications needed to run AI-Scientist with cloud-based models:

- **RAM**: 8GB
  - Required for basic operation and model API usage
  - Sufficient for running with cloud-based models

- **Storage**: 10GB
  - 2GB for base installation
  - 5GB for dependencies
  - 3GB for workspace and generated content

- **CPU**: 4 cores
  - Recommended minimum for concurrent operations
  - Suitable for basic research tasks

- **Python**: 3.8-3.11
  - Required for compatibility with dependencies
  - Latest patch version within range recommended

## Recommended Requirements

These specifications are recommended for optimal performance, especially when using local models:

- **RAM**: 16GB
  - Recommended for running local models
  - Provides better performance for concurrent operations
  - Required for larger research projects

- **Storage**: 20GB
  - 2GB for base installation
  - 8GB for dependencies
  - 5GB for local model files
  - 5GB for workspace and generated content

- **GPU**: NVIDIA with 8GB VRAM (for local models)
  - Required only if running local models
  - NVIDIA GPU recommended for compatibility
  - Minimum 8GB VRAM for standard model variants
  - Required for optimal performance with local LLMs

- **CUDA**: 11.8+
  - Required only if using GPU
  - Compatible with PyTorch and most ML frameworks
  - Latest version recommended within compatibility range

## Notes

- GPU requirements are optional and only necessary for running local models
- Cloud-based usage can run effectively on minimum specifications
- Storage requirements may vary based on the number of local models installed
- For development work, recommended specifications are strongly advised
