# Dockerfile
ARG COMPILER
ARG MATHLIB

FROM docker/compilers/${COMPILER}:latest as compiler
FROM docker/mathlibs/${MATHLIB}:latest as mathlib

# Combine stages
FROM compiler as final
COPY --from=mathlib / /

# Additional setup
RUN git clone --recursive https://github.com/pytorch/pytorch /pytorch

WORKDIR /pytorch

RUN mkdir build && cd build && \
    cmake .. -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX \
             -DUSE_CUDA=OFF -DUSE_CUDNN=OFF -DBUILD_CAFFE2_OPS=OFF \
             -DUSE_BLAS=${MATHLIB} && \
    make -j$(nproc)

RUN python3 setup.py install

# Test the installation
RUN python3 -c "import torch; print(torch.__version__)"

ENTRYPOINT ["python3"]

