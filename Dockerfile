FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Cài Python và pip
RUN apt-get update && apt-get install -y python3-pip

# Cài PyTorch & TensorFlow GPU (chọn phiên bản phù hợp)
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install --no-cache-dir tensorflow

WORKDIR /workspace
CMD ["/bin/bash"]
