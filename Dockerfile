# Use texlive/texlive as the base image which already has TeX Live installed
FROM texlive/texlive:latest

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
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
    git \
    && rm -rf /var/lib/apt/lists/*

# Download and install Python 3.11
RUN wget https://www.python.org/ftp/python/3.11.4/Python-3.11.4.tgz \
    && tar xzf Python-3.11.4.tgz \
    && cd Python-3.11.4 \
    && ./configure --enable-optimizations \
    && make altinstall \
    && cd .. \
    && rm -rf Python-3.11.4 Python-3.11.4.tgz

# Update alternatives to set Python 3.11 as the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip3 pip3 /usr/local/bin/pip3.11 1

# Install Python packages
RUN pip3 install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir \
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
    python-dotenv \
    einops

# Clone and install NPEET
RUN git clone https://github.com/gregversteeg/NPEET.git \
    && cd NPEET \
    && pip3 install .

# Clone the AI-Scientist repository
RUN git clone https://github.com/SakanaAI/AI-Scientist.git

# Set working directory to AI-Scientist
WORKDIR /app/AI-Scientist

# Prepare NanoGPT data
RUN python3 data/enwik8/prepare.py \
    && python3 data/shakespeare_char/prepare.py \
    && python3 data/text8/prepare.py

# Set the default command to open a bash shell
CMD ["/bin/bash"]