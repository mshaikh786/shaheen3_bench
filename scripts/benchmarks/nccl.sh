#!/bin/bash

# Display usage information
show_usage() {
  echo "Usage: $0 --runtime <runtime> [--image-dir <path>]"
  echo "Options:"
  echo "  --runtime <runtime>            Container runtime to use (e.g., docker, singularity, apptainer)"
  echo "  --image-dir <path>             Path to check for the container image (optional)"
  echo "  --help                         Show this help message and exit"
}

# Parse command-line arguments
runtime=""
image_dir=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runtime)
      runtime="$2"
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

# Define image and paths
nccl_image="nvcr.io/nvidia/nccl-tests:latest"
workspace_path="/workspace"
all_reduce_perf_exec="$workspace_path/all_reduce_perf"

# Check if the image exists in the specified directory
if [[ -n "$image_dir" ]]; then
  echo "Checking for image in directory: $image_dir"
  if [[ ! -f "$image_dir/$(basename $nccl_image)" ]]; then
    echo "Error: Image not found in directory: $image_dir"
    exit 1
  fi
fi

# Run the NCCL benchmarks
case "$runtime" in
  docker)
    echo "Running NCCL benchmarks using Docker..."
    docker run --rm --gpus all -v "$PWD:$workspace_path" $nccl_image bash -c "\
      $all_reduce_perf_exec -b 8 -e 1024M -f 2 -g 1 -c 1 -n 50 -w 20 && \
      $all_reduce_perf_exec -b 1G -e 24G -f 2 -g 1 -c 1 -n 50 -w 20 -o all -d all"
    ;;
  singularity|apptainer)
    echo "Running NCCL benchmarks using $runtime..."
    $runtime exec --nv docker://$nccl_image bash -c "\
      $all_reduce_perf_exec -b 8 -e 1024M -f 2 -g 1 -c 1 -n 50 -w 20 && \
      $all_reduce_perf_exec -b 1G -e 24G -f 2 -g 1 -c 1 -n 50 -w 20 -o all -d all"
    ;;
  *)
    echo "Error: Unsupported container runtime '$runtime'."
    show_usage
    exit 1
    ;;
esac

echo "NCCL benchmarks completed successfully."
exit 0

