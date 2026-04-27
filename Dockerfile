FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ARG FLIGHTMARE_REF=master
ARG FLIGHTMARE_RENDERER_URL=https://github.com/uzh-rpg/flightmare/releases/download/0.0.5/RPG_Flightmare.tar.xz
ARG ZMQPP_REF=4.2.0

# NOTE: vLLM 0.11.2 will automatically upgrade PyTorch during pip install.
# This is required for RTX 5070 Ti (Blackwell, sm_120) support
# Base image provides PyTorch 2.5.1, but final runtime uses the vLLM-supported
# modern PyTorch version resolved from requirements.txt.

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_ALLOW_TF32_CUBLAS=1 \
    CUDA_MODULE_LOADING=LAZY \
    TORCH_CUDNN_V8_API_ENABLED=1 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    FLIGHTMARE_PATH=/opt/flightmare \
    FLIGHTMARE_UNITY_EXECUTABLE=/opt/flightmare/flightrender/RPG_Flightmare.x86_64

RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates wget xz-utils unzip \
      build-essential g++ gcc cmake ninja-build pkg-config \
      openmpi-bin libopenmpi-dev \
      libzmq3-dev \
      libopencv-dev libeigen3-dev libgoogle-glog-dev libprotobuf-dev protobuf-compiler \
      libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
      libgl1-mesa-glx libglu1-mesa xvfb libosmesa6 \
      libegl1 libgles2 libegl-mesa0 libglvnd0 libglx0 libopengl0 \
      libx11-6 libxcursor1 libxrandr2 libxi6 libxinerama1 libxxf86vm1 libxss1 libxtst6 \
      libgtk-3-0 libnss3 libasound2 libpulse0 libdbus-1-3 libfontconfig1 libfreetype6 \
      libvulkan1 mesa-vulkan-drivers vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu Jammy no longer ships libzmqpp-dev, but Flightmare still links zmqpp.
# Build the latest upstream zmqpp release against Jammy's libzmq3-dev.
RUN git clone --depth 1 --branch "${ZMQPP_REF}" https://github.com/zeromq/zmqpp.git /tmp/zmqpp && \
    make -C /tmp/zmqpp -j"$(nproc)" && \
    make -C /tmp/zmqpp install PREFIX=/usr && \
    ldconfig && \
    rm -rf /tmp/zmqpp

WORKDIR /workspace

COPY scripts/flightmare_bc/patch_flightmare_pybind.py /tmp/patch_flightmare_pybind.py

# Install dependencies first for Docker layer caching
COPY requirements.txt /workspace/requirements.txt
COPY requirements/ /workspace/requirements/
RUN pip install --upgrade pip && \
    pip install -r /workspace/requirements.txt && \
    pip install -r /workspace/requirements/eval.txt

# Flightmare's flightrl package pins legacy TensorFlow-era RL deps; this repo supplies RL.
RUN git clone --depth 1 --branch "${FLIGHTMARE_REF}" https://github.com/uzh-rpg/flightmare.git "${FLIGHTMARE_PATH}" && \
    curl -L "${FLIGHTMARE_RENDERER_URL}" -o /tmp/RPG_Flightmare.tar.xz && \
    mkdir -p "${FLIGHTMARE_PATH}/flightrender" && \
    tar -xJf /tmp/RPG_Flightmare.tar.xz -C "${FLIGHTMARE_PATH}/flightrender" --strip-components=1 && \
    rm /tmp/RPG_Flightmare.tar.xz && \
    chmod +x "${FLIGHTMARE_UNITY_EXECUTABLE}" && \
    sed -i 's/option(BUILD_TESTS "Building the tests" ON)/option(BUILD_TESTS "Building the tests" OFF)/' "${FLIGHTMARE_PATH}/flightlib/CMakeLists.txt" && \
    sed -i 's/option(BUILD_UNITY_BRIDGE_TESTS "Building the Unity Bridge tests" ON)/option(BUILD_UNITY_BRIDGE_TESTS "Building the Unity Bridge tests" OFF)/' "${FLIGHTMARE_PATH}/flightlib/CMakeLists.txt" && \
    python /tmp/patch_flightmare_pybind.py && \
    pip install --no-deps "${FLIGHTMARE_PATH}/flightlib"

# Copy project code
COPY . /workspace

# Install the package in editable mode
RUN pip install -e .

# Keep container running for interactive CLI access
CMD ["tail", "-f", "/dev/null"]
