# Model Configuration Guide

This guide provides detailed instructions for setting up and configuring different model options in AI-Scientist.

## Cloud API Models

### 1. OpenAI Models
- **Required Resources**
  - RAM: 4GB minimum
  - Storage: 2GB for dependencies
  - Internet connection required
- **Available Models**
  - gpt-4-turbo (recommended)
  - gpt-3.5-turbo
- **Setup Instructions**
  1. Obtain API key from [OpenAI Platform](https://platform.openai.com)
  2. Set environment variable:
     ```bash
     export OPENAI_API_KEY="your-key-here"
     ```

### 2. Google Gemini Pro
- **Required Resources**
  - RAM: 4GB minimum
  - Storage: 2GB for dependencies
  - Internet connection required
- **Available Models**
  - gemini-pro
- **Setup Instructions**
  1. Set up Google Cloud project
  2. Enable Vertex AI API
  3. Set environment variables:
     ```bash
     export CLOUD_ML_REGION="your-region"
     export VERTEXAI_PROJECT="your-project-id"
     ```

### 3. Anthropic Claude
- **Required Resources**
  - RAM: 4GB minimum
  - Storage: 2GB for dependencies
  - Internet connection required
- **Available Models**
  - claude-3-sonnet (recommended)
  - claude-3-opus
- **Setup Instructions**
  1. Obtain API key from Anthropic
  2. Set environment variable:
     ```bash
     export ANTHROPIC_API_KEY="your-key-here"
     ```

### 4. DeepSeek Coder
- **Required Resources**
  - RAM: 4GB minimum
  - Storage: 2GB for dependencies
  - Internet connection required
- **Available Models**
  - deepseek-coder-v2
- **Setup Instructions**
  1. Obtain API key from DeepSeek
  2. Set environment variable:
     ```bash
     export DEEPSEEK_API_KEY="your-key-here"
     ```

## Local Models (via Ollama)

### Prerequisites
- Ollama installed ([Installation Guide](https://ollama.ai/download))
- System requirements vary by model size

### 1. LLaMA Models
- **Required Resources**
  - RAM: 8GB minimum (16GB recommended)
  - Storage: 5-15GB depending on model size
  - GPU: Optional, 8GB VRAM recommended
- **Available Models**
  - llama3.2:1b (minimum requirements)
  - llama3.2:7b (recommended)
  - llama3.3:7b (recommended)
- **Setup Instructions**
  1. Install Ollama
  2. Pull model:
     ```bash
     ollama pull llama3.2:7b
     ```
  3. Start Ollama server:
     ```bash
     ollama serve
     ```

### 2. Mistral Models
- **Required Resources**
  - RAM: 6GB minimum (12GB recommended)
  - Storage: 4-8GB
  - GPU: Optional, 8GB VRAM recommended
- **Available Models**
  - mistral:7b
- **Setup Instructions**
  1. Install Ollama
  2. Pull model:
     ```bash
     ollama pull mistral:7b
     ```
  3. Start Ollama server:
     ```bash
     ollama serve
     ```

### 3. Code LLaMA
- **Required Resources**
  - RAM: 8GB minimum (16GB recommended)
  - Storage: 5-15GB
  - GPU: Optional, 8GB VRAM recommended
- **Available Models**
  - codellama:7b
- **Setup Instructions**
  1. Install Ollama
  2. Pull model:
     ```bash
     ollama pull codellama:7b
     ```
  3. Start Ollama server:
     ```bash
     ollama serve
     ```

## Troubleshooting

### Common Issues

1. **Insufficient Memory**
   - Try using smaller model variants
   - Close unnecessary applications
   - For local models, use CPU-only mode if GPU memory is insufficient

2. **API Rate Limits**
   - Implement exponential backoff (built into our client)
   - Consider using multiple API keys
   - Monitor usage through provider dashboards

3. **Local Model Performance**
   - Use GPU acceleration when available
   - Consider cloud API models for better performance
   - Try smaller model variants if experiencing issues

### Getting Help

If you encounter issues not covered here:
1. Check the [GitHub Issues](https://github.com/SakanaAI/AI-Scientist/issues)
2. Review error messages in logs
3. Create a new issue with detailed system information
