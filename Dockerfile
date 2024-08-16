# Use Python 3.11 as the base image
FROM python:3.11-bullseye

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies including texlive-full
RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    texlive-full \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install Python packages
RUN pip install --no-cache-dir \
    anthropic \
    aider-chat \
    backoff \
    openai \
    matplotlib \
    pypdf \
    pymupdf4llm \
    torch \
    numpy \
    transformers \
    datasets \
    tiktoken \
    wandb \
    tqdm \
    scikit-learn \
    einops

# Clone and install NPEET
RUN git clone https://github.com/gregversteeg/NPEET.git \
    && cd NPEET \
    && pip install .

# Clone the AI-Scientist repository
RUN git clone https://github.com/SakanaAI/AI-Scientist.git

# Set working directory to AI-Scientist
WORKDIR /app/AI-Scientist

# Prepare NanoGPT data
RUN python data/enwik8/prepare.py \
    && python data/shakespeare_char/prepare.py \
    && python data/text8/prepare.py

# Set up baseline runs
RUN for dir in templates/*/; do \
    if [ -f "${dir}experiment.py" ]; then \
        cd "$dir" && python experiment.py --out_dir run_0 && python plot.py && cd /app/AI-Scientist; \
    fi \
done

# Set the default command to open a bash shell
CMD ["/bin/bash"]