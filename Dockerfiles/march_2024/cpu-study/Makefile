# Default target
.DEFAULT_GOAL := build

# Define valid compilers and math libraries
VALID_COMPILERS := gcc cray aocc
VALID_MATH_LIBS := mkl libsci blis openblas

# Define Dockerfile paths
DOCKERFILE_PATHS.gcc.mkl := docker/gcc/mkl/Dockerfile
DOCKERFILE_PATHS.gcc.libsci := docker/gcc/libsci/Dockerfile
DOCKERFILE_PATHS.gcc.blis := docker/gcc/blis/Dockerfile
DOCKERFILE_PATHS.gcc.openblas := docker/gcc/openblas/Dockerfile
DOCKERFILE_PATHS.cray.mkl := docker/cray/mkl/Dockerfile
DOCKERFILE_PATHS.cray.libsci := docker/cray/libsci/Dockerfile
DOCKERFILE_PATHS.cray.blis := docker/cray/blis/Dockerfile
DOCKERFILE_PATHS.cray.openblas := docker/cray/openblas/Dockerfile
DOCKERFILE_PATHS.aocc.mkl := docker/aocc/mkl/Dockerfile
DOCKERFILE_PATHS.aocc.libsci := docker/aocc/libsci/Dockerfile
DOCKERFILE_PATHS.aocc.blis := docker/aocc/blis/Dockerfile
DOCKERFILE_PATHS.aocc.openblas := docker/aocc/openblas/Dockerfile

# Define image names
IMAGE_NAME.gcc.mkl := pytorch-gcc-mkl:latest
IMAGE_NAME.gcc.libsci := pytorch-gcc-libsci:latest
IMAGE_NAME.gcc.blis := pytorch-gcc-blis:latest
IMAGE_NAME.gcc.openblas := pytorch-gcc-openblas:latest
IMAGE_NAME.cray.mkl := pytorch-cray-mkl:latest
IMAGE_NAME.cray.libsci := pytorch-cray-libsci:latest
IMAGE_NAME.cray.blis := pytorch-cray-blis:latest
IMAGE_NAME.cray.openblas := pytorch-cray-openblas:latest
IMAGE_NAME.aocc.mkl := pytorch-aocc-mkl:latest
IMAGE_NAME.aocc.libsci := pytorch-aocc-libsci:latest
IMAGE_NAME.aocc.blis := pytorch-aocc-blis:latest
IMAGE_NAME.aocc.openblas := pytorch-aocc-openblas:latest

# Check the USE_COMPILER and USE_MATH_LIB environment variables and validate
check_vars:
ifndef USE_COMPILER
	$(error USE_COMPILER is undefined. Please set it to one of $(VALID_COMPILERS))
endif
ifeq (,$(findstring $(USE_COMPILER),$(VALID_COMPILERS)))
	$(error Invalid USE_COMPILER value. Valid values are: $(VALID_COMPILERS))
endif
ifndef USE_MATH_LIB
	$(error USE_MATH_LIB is undefined. Please set it to one of $(VALID_MATH_LIBS))
endif
ifeq (,$(findstring $(USE_MATH_LIB),$(VALID_MATH_LIBS)))
	$(error Invalid USE_MATH_LIB value. Valid values are: $(VALID_MATH_LIBS))
endif

# Build the Docker image based on the selected compiler and math library
build: check_vars
	docker build -t $(IMAGE_NAME.$(USE_COMPILER).$(USE_MATH_LIB)) -f $(DOCKERFILE_PATHS.$(USE_COMPILER).$(USE_MATH_LIB)) .

.PHONY: check_vars build

