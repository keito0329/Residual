FROM nvidia/cuda:11.8.0-devel-ubuntu20.04
  
# Install Python 3.10.16
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

  
# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Working directory
WORKDIR /app  
  
# Install PyTorch (CUDA-enabled)
# Pin to the Python 3.8-compatible PyTorch build
# Pin typing-extensions to a Python 3.8-compatible version
RUN pip install --no-cache-dir "typing-extensions<4.9"

# PyTorch (Python 3.8 + CUDA 11.8 compatible)
RUN pip install --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2 \
    torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

  
# Other dependencies
COPY requirements.txt .  
RUN pip install --no-cache-dir -r requirements.txt  
  
# Source code and environment variables
COPY . .  
