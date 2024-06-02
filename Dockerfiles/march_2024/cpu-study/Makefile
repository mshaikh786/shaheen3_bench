# Default target
.DEFAULT_GOAL := build

# Define valid compilers and math libraries
VALID_COMPILERS := gcc cray aocc
VALID_MATH_LIBS := mkl libsci blis openblas

# Check the COMPILER and MATHLIB environment variables and validate
check_vars:
	@if [ -z "${COMPILER}" ]; then \
		echo "COMPILER is undefined. Please set it to one of ${VALID_COMPILERS}"; \
		exit 1; \
	fi
	@if [ -z "${MATHLIB}" ]; then \
		echo "MATHLIB is undefined. Please set it to one of ${VALID_MATH_LIBS}"; \
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

# Build the Docker image based on the selected compiler and math library
build: check_vars
	docker build --build-arg COMPILER=${COMPILER} --build-arg MATHLIB=${MATHLIB} -t pytorch-${COMPILER}-${MATHLIB}:latest .

.PHONY: check_vars build
