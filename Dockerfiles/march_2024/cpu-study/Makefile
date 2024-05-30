# Default target
.DEFAULT_GOAL := build

# Define valid compilers and math libraries
VALID_COMPILERS := gcc cray aocc
VALID_MATH_LIBS := mkl libsci blis openblas

# Check the USE_COMPILER and USE_MATH_LIB environment variables and validate
check_vars:
ifndef COMPILER
	$(error COMPILER is undefined. Please set it to one of $(VALID_COMPILERS))
endif
ifeq (,$(findstring $(COMPILER),$(VALID_COMPILERS)))
	$(error Invalid COMPILER value. Valid values are: $(VALID_COMPILERS))
endif
ifndef MATHLIB
	$(error MATHLIB is undefined. Please set it to one of $(VALID_MATH_LIBS))
endif
ifeq (,$(findstring $(MATHLIB),$(VALID_MATH_LIBS)))
	$(error Invalid MATHLIB value. Valid values are: $(VALID_MATH_LIBS))
endif

# Build the Docker image based on the selected compiler and math library
build: check_vars
	docker build --build-arg COMPILER=$(COMPILER) --build-arg MATHLIB=$(MATHLIB) -t pytorch-$(COMPILER)-$(MATHLIB):latest .

.PHONY: check_vars build

