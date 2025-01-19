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
cuda_image="nvcr.io/nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04"
workspace_path="/workspace"
device_query_exec="$workspace_path/Samples/1_Utilities/deviceQuery/deviceQuery"
bandwidth_test_exec="$workspace_path/Samples/1_Utilities/bandwidthTest/bandwidthTest"

# Check if the image exists in the specified directory
if [[ -n "$image_dir" ]]; then
  echo "Checking for image in directory: $image_dir"
  if [[ ! -f "$image_dir/$(basename $cuda_image)" ]]; then
    echo "Error: Image not found in directory: $image_dir"
    exit 1
  fi
fi

# Run the CUDA samples benchmark
case "$runtime" in
  docker)
    echo "Running CUDA Samples benchmark using Docker..."
    docker run --rm --gpus all -v "$PWD:$workspace_path" $cuda_image bash -c "\
      $device_query_exec && \
      $bandwidth_test_exec"
    ;;
  singularity|apptainer)
    echo "Running CUDA Samples benchmark using $runtime..."
    $runtime exec --nv docker://$cuda_image bash -c "\
      $device_query_exec && \
      $bandwidth_test_exec"
    ;;
  *)
    echo "Error: Unsupported container runtime '$runtime'."
    show_usage
    exit 1
    ;;
esac

echo "CUDA Samples benchmark completed successfully."
exit 0

