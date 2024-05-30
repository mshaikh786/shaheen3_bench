# Use an appropriate base image
FROM ubuntu:20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    cmake \
    python3-dev \
    python3-venv \
    python3-pip \
    wget \
    curl \
    ca-certificates \
    libopenmpi-dev \
    libomp-dev \
    libopenblas-dev

# Set environment variables for OpenBLAS
ENV OPENBLASROOT=/usr
ENV LD_LIBRARY_PATH=$OPENBLASROOT/lib:$LD_LIBRARY_PATH

# Install Python dependencies
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install numpy pyyaml setuptools cmake cffi typing_extensions future six requests dataclasses

# Clone PyTorch repository
RUN git clone --recursive https://github.com/pytorch/pytorch /pytorch

# Set up environment for building with OpenBLAS
ENV USE_CUDA=0
ENV USE_CUDNN=0
ENV USE_MKLDNN=0
ENV USE_DISTRIBUTED=1

# Build PyTorch
WORKDIR /pytorch
RUN mkdir build && cd build && \
    cmake .. -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
             -DUSE_CUDA=OFF -DUSE_CUDNN=OFF -DBUILD_CAFFE2_OPS=OFF \
             -DUSE_BLAS=open -DBLAS_INCLUDE_DIR=$OPENBLASROOT/include -DBLAS_LIBRARY=$OPENBLASROOT/lib/libopenblas.a && \
    make -j$(nproc)

# Install PyTorch
RUN python3 setup.py install

# Test the installation
RUN python3 -c "import torch; print(torch.__version__)"

# Set the entrypoint to Python shell
ENTRYPOINT ["python3"]

