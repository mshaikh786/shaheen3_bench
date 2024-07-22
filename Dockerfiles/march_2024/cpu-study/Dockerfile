# Dockerfile

ARG COMPILER
ARG MATHLIB
ARG DNNLIB
ARG USE_CUDA=off

# Stage 1: Compiler
FROM docker/compilers/${COMPILER}:latest AS compiler

# Stage 2: Math Library
FROM docker/mathlibs/${MATHLIB}:latest AS mathlib

# Stage 3: DNN Library
FROM docker/dnn/${DNNLIB}:latest AS dnnlib

# Conditional CUDA stage
FROM docker/base:latest AS base

# Combine stages
FROM compiler AS final

COPY --from=mathlib / /
COPY --from=dnnlib / /
COPY --from=base / /
COPY --from=compiler / /

# Install Ninja
RUN apt-get update && apt-get install -y ninja-build

# Additional setup

RUN git clone --depth 1 https://github.com/pytorch/pytorch /pytorch;

WORKDIR /pytorch

RUN git config --global http.postBuffer 524288000 && \
    git submodule update --init --recursive

# Install the necessary dependencies
RUN python3 -m pip install -r requirements.txt

# Install PyTorch using the develop method
RUN python3 setup.py develop

# Set the library path to include PyTorch libraries
ENV LD_LIBRARY_PATH /usr/lib:/pytorch/build/lib:/pytorch/torch/lib:$LD_LIBRARY_PATH
ENV PATH /pytorch/build/lib:/pytorch/torch/lib:$PATH

# Test the installation from a different directory
WORKDIR /
RUN python3 -c "import torch; print(torch.__config__.show())"

CMD ["/bin/bash"]
