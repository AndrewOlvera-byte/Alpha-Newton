FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_ALLOW_TF32_CUBLAS=1 \
    CUDA_MODULE_LOADING=LAZY \
    TORCH_CUDNN_V8_API_ENABLED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates \
      build-essential g++ gcc \
      libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install dependencies first for Docker layer caching
COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip && pip install -r /workspace/requirements.txt

# Copy project code
COPY . /workspace

# Install the package in editable mode
RUN pip install -e .

# Keep container running for interactive CLI access
CMD ["tail", "-f", "/dev/null"]
