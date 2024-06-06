# Dockerfile

ARG COMPILER
ARG MATHLIB
ARG DNNLIB

# Stage 1: Compiler
FROM docker/compilers/${COMPILER}:latest as compiler

# Stage 2: Math Library
FROM docker/mathlibs/${MATHLIB}:latest as mathlib

# Stage 3: DNN Library
FROM docker/dnn/${DNNLIB}:latest as dnnlib

# Combine stages
FROM compiler as final
COPY --from=mathlib / /
COPY --from=dnnlib / /

# Install Ninja
RUN apt-get update && apt-get install -y ninja-build

# Additional setup
RUN git config --global http.postBuffer 1048576000 && \
    until git clone --depth 1 https://github.com/pytorch/pytorch /pytorch; do \
    echo "Retrying"; \
    sleep 5; \
    done

WORKDIR /pytorch

# Install the necessary dependencies
RUN python3 -m pip install -r requirements.txt

# Ensure a clean build directory and use Ninja as the generator
RUN rm -rf build && mkdir build && cd build && \
    cmake -GNinja .. -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX \
             -DUSE_CUDA=OFF -DUSE_CUDNN=OFF -DBUILD_CAFFE2_OPS=OFF \
             -DUSE_${MATHLIB}=ON -DUSE_${DNNLIB}=ON && \
    ninja

# Install PyTorch using the develop method
RUN python3 setup.py develop

# Set the library path to include PyTorch libraries
ENV LD_LIBRARY_PATH /pytorch/torch/lib:$LD_LIBRARY_PATH

# Test the installation from a different directory
WORKDIR /
RUN python3 -c "import torch; print(torch.__version__)"

ENTRYPOINT ["python3"]
