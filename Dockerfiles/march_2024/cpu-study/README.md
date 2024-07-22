# Dynamic Docker Image Builder for PyTorch with Custom Compilers and Math Libraries

This project provides a framework for building Docker images for PyTorch using various compilers and math libraries. The build process dynamically generates Dockerfiles by combining base configurations with specified compiler and math library configurations, making the setup flexible and maintainable.

## Directory Structure

```
├── docker
│ ├── base
│ │ └── Dockerfile.base
│ ├── compilers
│ │ ├── gcc
│ │ │ └── Dockerfile.gcc
│ │ ├── cray
│ │ │ └── Dockerfile.cray
│ │ └── aocc
│ │   └── Dockerfile.aocc
│ ├── mathlibs
│ │ ├── mkl
│ │ │ └── Dockerfile.mkl
│ │ ├── libsci
│ │ │ └── Dockerfile.libsci
│ │ ├── blis
│ │ │ └── Dockerfile.blis
│ │ └── openblas
│ │   └── Dockerfile.openblas
│ ├── dnn
│ │ ├── onednn
│ │ │ └── Dockerfile.onednn
│ │ ├── zendnn
│ │ │ └── Dockerfile.zendnn
│ │ └── fbgemm
│ │   └── Dockerfile.fbgemm
├── Dockerfile
└── Makefile
```

## Getting Started

### Prerequisites

- Docker installed on your system.
- Internet connection to download dependencies and source code.

### Build Process

The build process consists of the following steps:

1. Define the base Dockerfile.
2. Define Dockerfiles for each compiler and math library.
3. Use a script to dynamically combine these Dockerfiles into a final Dockerfile.
4. Build the Docker image using the final Dockerfile.

### Components
- Base Dockerfile (docker/base/Dockerfile.base): Contains the base configuration and common dependencies.
- Compiler Dockerfiles (docker/compilers/*): Contains configurations for different compilers.
- Math Library Dockerfiles (docker/mathlibs/*): Contains configurations for different math libraries.
- DNN Dockerfiles (docker/dnn/*): Contains configurations for different dnn libraries

### Usage
1. **Set Environment Variables:**

   Define the compiler and math library you want to use. Valid values are:

    - **Compilers**: `gcc`, `cray`, `aocc`
    - **Math Libraries**: `mkl`, `libsci`, `blis`, `openblas`
    - **DNN Libraries**: `onednn`, `zendnn`, `fbgemm`


2. **Run the Build Script:**

    Use the provided Makefile to run the build script.
   #### Example
    ```
    COMPILER=gcc MATHLIB=mkl DNNLIB=onednn make build_main
    ```