FROM nvidia/cuda:11.8.0-devel-ubuntu20.04
  
# Python 3.10.16のインストール  
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

  
# Python 3.10をデフォルトに設定  
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# 作業ディレクトリ  
WORKDIR /app  
  
# PyTorch（CUDA対応）のインストール  
# PyTorch（Python 3.8対応版を明示指定）
# typing-extensions を Python 3.8 対応版に固定
RUN pip install --no-cache-dir "typing-extensions<4.9"

# PyTorch（Python 3.8 + CUDA 11.8 対応）
RUN pip install --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2 \
    torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

  
# その他の依存関係  
COPY requirements.txt .  
RUN pip install --no-cache-dir -r requirements.txt  
  
# ソースコードと環境変数  
COPY . .  