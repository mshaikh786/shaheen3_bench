# Default target
.DEFAULT_GOAL := build

# Default CUDA usage
USE_CUDA ?= off

# Define valid compilers, math libraries, and DNN libraries
VALID_COMPILERS := gcc cray aocc
VALID_MATH_LIBS := mkl libsci blis openblas
VALID_DNN_LIBS := onednn zendnn fbgemm

# Compute the tag suffix based on USE_CUDA
ifeq ($(USE_CUDA),on)
    TAG_SUFFIX := cuda
else
    TAG_SUFFIX := cpu
endif

# Check the COMPILER, MATHLIB, and DNNLIB environment variables and validate
check_vars:
	@if [ -z "${COMPILER}" ]; then \
		echo "COMPILER is undefined. Please set it to one of ${VALID_COMPILERS}"; \
		exit 1; \
	fi
	@if [ -z "${MATHLIB}" ]; then \
		echo "MATHLIB is undefined. Please set it to one of ${VALID_MATH_LIBS}"; \
		exit 1; \
	fi
	@if [ -z "${DNNLIB}" ]; then \
		echo "DNNLIB is undefined. Please set it to one of ${VALID_DNN_LIBS}"; \
		exit 1; \
	fi
	@if ! echo "${VALID_COMPILERS}" | grep -qw "${COMPILER}"; then \
		echo "Invalid COMPILER value. Valid values are: ${VALID_COMPILERS}"; \
		exit 1; \
	fi
	@if ! echo "${VALID_MATH_LIBS}" | grep -qw "${MATHLIB}"; then \
		echo "Invalid MATHLIB value. Valid values are: ${VALID_MATH_LIBS}"; \
		exit 1; \
	fi
	@if ! echo "${VALID_DNN_LIBS}" | grep -qw "${DNNLIB}"; then \
		echo "Invalid DNNLIB value. Valid values are: ${VALID_DNN_LIBS}"; \
		exit 1; \
	fi

# Build the compiler image
build_compiler: check_vars
	docker build -f docker/compilers/${COMPILER}/Dockerfile.${COMPILER} -t docker/compilers/${COMPILER}:latest .

# Build the math library image
build_mathlib: check_vars
	docker build -f docker/mathlibs/${MATHLIB}/Dockerfile.${MATHLIB} -t docker/mathlibs/${MATHLIB}:latest .

# Build the DNN library image
build_dnnlib: check_vars
	docker build -f docker/dnn/${DNNLIB}/Dockerfile.${DNNLIB} -t docker/dnn/${DNNLIB}:latest .

# Build CUDA Docker image if USE_CUDA is on
build_base:
	@if [ "$(USE_CUDA)" = "on" ]; then \
		echo "Building with CUDA support..."; \
		docker build -f docker/base/Dockerfile.cuda -t docker/base:latest .; \
	else \
		echo "Building without CUDA support..."; \
		docker build -f docker/base/Dockerfile.base -t docker/base:latest .; \
	fi

# Build the main Docker image based on the selected compiler, math library, and DNN library
build_main: build_compiler build_mathlib build_dnnlib
	docker build --build-arg COMPILER=${COMPILER} --build-arg MATHLIB=${MATHLIB} --build-arg DNNLIB=${DNNLIB} -f Dockerfile -t pytorch-${COMPILER}-${MATHLIB}-${DNNLIB}:${TAG_SUFFIX} .

.PHONY: check_vars build_compiler build_mathlib build_dnnlib build_main build_base
