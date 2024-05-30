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

# Install Intel MKL
RUN wget -q https://registrationcenter-download.intel.com/akdlm/irc_nas/tec/17415/l_dnnl_1.6.4.915.tar.gz -O oneDNN.tar.gz && \
    tar -xzf oneDNN.tar.gz && \
    cd l_dnnl_* && \
    ./install.sh && \
    cd .. && \
    rm -rf l_dnnl_*

# Set environment variables for MKL
ENV MKLROOT=/opt/intel/oneapi/mkl/latest
ENV LD_LIBRARY_PATH=$MKLROOT/lib/intel64:$LD_LIBRARY_PATH

# Install Python dependencies
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install numpy pyyaml setuptools cmake cffi typing_extensions future six requests dataclasses

# Clone PyTorch repository
RUN git clone --recursive https://github.com/pytorch/pytorch /pytorch

# Set up environment for building with MKL
ENV USE_CUDA=0
ENV USE_CUDNN=0
ENV USE_MKLDNN=1
ENV USE_DISTRIBUTED=1
ENV MKLDNNROOT=$MKLROOT

# Build PyTorch
WORKDIR /pytorch
RUN mkdir build && cd build && \
    cmake .. -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX \
             -DUSE_CUDA=OFF -DUSE_CUDNN=OFF -DBUILD_CAFFE2_OPS=OFF \
             -DUSE_MKLDNN=ON -DMKL_INCLUDE_DIR=$MKLROOT/include -DMKL_LIBRARY=$MKLROOT/lib/intel64 && \
    make -j$(nproc)

# Install PyTorch
RUN python3 setup.py install

# Test the installation
RUN python3 -c "import torch; print(torch.__version__)"

# Set the entrypoint to Python shell
ENTRYPOINT ["python3"]

