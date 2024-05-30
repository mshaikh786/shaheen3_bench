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
    libomp-dev

# Install AOCC
RUN wget -qO- http://developer.amd.com/wordpress/media/files/aocc-compiler-2.3.0_Ubuntu1804.tar | tar xvf - && \
    cd aocc-compiler-2.3.0_Ubuntu1804 && \
    ./install.sh && \
    cd .. && \
    rm -rf aocc-compiler-2.3.0_Ubuntu1804

# Set environment variables for AOCC
ENV CC=/opt/AMD/aocc-compiler-2.3.0/bin/clang
ENV CXX=/opt/AMD/aocc-compiler-2.3.0/bin/clang++

# Install BLIS
RUN git clone https://github.com/flame/blis /blis && \
    cd /blis && \
    ./configure auto && \
    make -j$(nproc) && \
    make install && \
    cd .. && \
    rm -rf /blis

# Set environment variables for BLIS
ENV BLISROOT=/usr/local
ENV LD_LIBRARY_PATH=$BLISROOT/lib:$LD_LIBRARY_PATH

# Install Python dependencies
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install numpy pyyaml setuptools cmake cffi typing_extensions future six requests dataclasses

# Clone PyTorch repository
RUN git clone --recursive https://github.com/pytorch/pytorch /pytorch

# Set up environment for building with BLIS
ENV USE_CUDA=0
ENV USE_CUDNN=0
ENV USE_MKLDNN=0
ENV USE_DISTRIBUTED=1

# Build PyTorch
WORKDIR /pytorch
RUN mkdir build && cd build && \
    cmake .. -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX \
             -DUSE_CUDA=OFF -DUSE_CUDNN=OFF -DBUILD_CAFFE2_OPS=OFF \
             -DUSE_BLAS=blis -DBLAS_INCLUDE_DIR=$BLISROOT/include -DBLAS_LIBRARY=$BLISROOT/lib/libblis.a && \
    make -j$(nproc)

# Install PyTorch
RUN python3 setup.py install

# Test the installation
RUN python3 -c "import torch; print(torch.__version__)"

# Set the entrypoint to Python shell
ENTRYPOINT ["python3"]

