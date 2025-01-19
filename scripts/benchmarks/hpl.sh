#!/bin/bash

# Display usage information
show_usage() {
  echo "Usage: $0 --runtime <runtime> --hpl-dat <path> [--image-dir <path>]"
  echo "Options:"
  echo "  --runtime <runtime>            Container runtime to use (e.g., docker, singularity, apptainer)"
  echo "  --hpl-dat <path>               Path to the HPL.dat configuration file"
  echo "  --image-dir <path>             Path to check for the container image (optional)"
  echo "  --help                         Show this help message and exit"
}

# Parse command-line arguments
runtime=""
hpl_dat=""
image_dir=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runtime)
      runtime="$2"
      shift 2
      ;;
    --hpl-dat)
      hpl_dat="$2"
      shift 2
      ;;
    --image-dir)
      image_dir="$2"
      shift 2
      ;;
    --help)
      show_usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      show_usage
      exit 1
      ;;
  esac
done

# Validate required arguments
if [[ -z "$runtime" ]]; then
  echo "Error: --runtime is required."
  show_usage
  exit 1
fi

if [[ -z "$hpl_dat" ]]; then
  echo "Error: --hpl-dat is required."
  show_usage
  exit 1
fi

# Define image and paths
hpc_bench_image="nvcr.io/nvidia/hpc-benchmarks:latest"
hpl_executable="/opt/hpc-benchmarks/HPL/bin/xhpl"

# Check if the image exists in the specified directory
if [[ -n "$image_dir" ]]; then
  echo "Checking for image in directory: $image_dir"
  if [[ ! -f "$image_dir/$(basename $hpc_bench_image)" ]]; then
    echo "Error: Image not found in directory: $image_dir"
    exit 1
  fi
fi

# Check if HPL.dat file exists
if [[ ! -f "$hpl_dat" ]]; then
  echo "Error: HPL.dat file not found at $hpl_dat"
  exit 1
fi

# Run the HPL benchmark
case "$runtime" in
  docker)
    echo "Running HPL benchmark using Docker..."
    docker run --rm --gpus all -v "$PWD:$PWD" -w "$PWD" $hpc_bench_image bash -c "\
      cp $hpl_dat /workspace/HPL.dat && \
      cd /workspace && \
      $hpl_executable"
    ;;
  singularity|apptainer)
    echo "Running HPL benchmark using $runtime..."
    $runtime exec --nv docker://$hpc_bench_image bash -c "\
      cp $hpl_dat /workspace/HPL.dat && \
      cd /workspace && \
      $hpl_executable"
    ;;
  *)
    echo "Error: Unsupported container runtime '$runtime'."
    show_usage
    exit 1
    ;;
esac

echo "HPL benchmark completed successfully."
exit 0

