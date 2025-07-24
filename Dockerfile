FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Cài Python 3.10, pip, libGL và các thư viện cần thiết
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3-dev python-is-python3 \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Cập nhật pip
RUN pip3 install --upgrade pip

# Cài PyTorch + TensorFlow GPU
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir tensorflow==2.15.0

WORKDIR /workspace

CMD ["/bin/bash"]
