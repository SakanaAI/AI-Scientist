# Use Python 3.11 as the base image
FROM python:3.11-bullseye

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies including texlive-full
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget=1.21-1+deb11u1 \
    git=1:2.30.2-1+deb11u2 \
    build-essential=12.9 \
    libssl-dev=1.1.1w-0+deb11u1 \
    zlib1g-dev=1:1.2.11.dfsg-2+deb11u2 \
    libbz2-dev=1.0.8-4 \
    libreadline-dev=8.1-1 \
    libsqlite3-dev=3.34.1-3 \
    libncursesw5-dev=6.2+20201114-2+deb11u2 \
    xz-utils=5.2.5-2.1~deb11u1 \
    tk-dev=8.6.11+1 \
    libxml2-dev=2.9.10+dfsg-6.7+deb11u4 \
    libxmlsec1-dev=1.2.31-1 \
    libffi-dev=3.3-6 \
    liblzma-dev=5.2.5-2.1~deb11u1 \
    texlive-full=2020.20210202-3 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip==24.2

# Install Python packages
RUN pip install --no-cache-dir \
    anthropic==0.34.0 \
    aider-chat==0.50.1 \
    backoff==2.2.1 \
    openai==1.40.6 \
    matplotlib==3.9.2 \
    pypdf==4.3.1 \
    pymupdf4llm==0.0.10 \
    torch==2.4.0 \
    numpy==1.26.4 \
    transformers==4.44.0 \
    datasets==2.21.0 \
    tiktoken==0.7.0 \
    wandb==0.17.7 \
    tqdm==4.66.5 \
    scikit-learn==1.5.1 \
    einops==0.8.0

# Clone and install NPEET with a specific commit
RUN git clone https://github.com/gregversteeg/NPEET.git
WORKDIR /app/NPEET
RUN git checkout 8b0d9485423f74e5eb199324cf362765596538d3 \
    && pip install .

# Clone the AI-Scientist repository
WORKDIR /app
RUN git clone https://github.com/SakanaAI/AI-Scientist.git

# Set working directory to AI-Scientist
WORKDIR /app/AI-Scientist

# Prepare NanoGPT data
RUN python data/enwik8/prepare.py && \
    python data/shakespeare_char/prepare.py && \
    python data/text8/prepare.py

# Set up baseline runs
RUN for dir in templates/*/; do \
    if [ -f "${dir}experiment.py" ]; then \
        cd "${dir}" || continue; \
        python experiment.py --out_dir run_0 && \
        python plot.py; \
        cd /app/AI-Scientist || exit; \
    fi \
done

# Create entrypoint script
RUN printf '#!/bin/bash\n\
python launch_scientist.py "$@"\n' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Set the default command to an empty array
CMD []