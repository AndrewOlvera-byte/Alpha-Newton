FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# NOTE: vLLM 0.13.0 will automatically upgrade PyTorch to 2.9.0 during pip install
# This is required for RTX 5070 Ti (Blackwell, sm_120) support
# Base image provides PyTorch 2.5.1, but final runtime will use 2.9.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_ALLOW_TF32_CUBLAS=1 \
    CUDA_MODULE_LOADING=LAZY \
    TORCH_CUDNN_V8_API_ENABLED=1 \
    MUJOCO_GL=osmesa \
    PYOPENGL_PLATFORM=osmesa

RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates \
      build-essential g++ gcc \
      libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
      libgl1-mesa-glx xvfb libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install dependencies first for Docker layer caching
COPY requirements.txt /workspace/requirements.txt
COPY requirements/ /workspace/requirements/
RUN pip install --upgrade pip && \
    pip install -r /workspace/requirements.txt && \
    pip install -r /workspace/requirements/eval.txt

# Copy project code
COPY . /workspace

# Install the package in editable mode
RUN pip install -e .

# Keep container running for interactive CLI access
CMD ["tail", "-f", "/dev/null"]
