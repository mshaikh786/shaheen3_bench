#!/bin/bash

# Default images for benchmarks
declare -A DEFAULT_IMAGES
DEFAULT_IMAGES=(
  [system_insights]="nvcr.io/nvidia/system:insights-latest",
  [cuda_samples]="nvcr.io/nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04"
)

# Display usage information
show_usage() {
  echo "Usage: $0 --benchmark <benchmark_name> [--runtime <runtime>] [--image-dir <path>]"
  echo "Options:"
  echo "  --benchmark <benchmark_name>   Name of the benchmark to download image for (e.g., system_insights, cuda_samples)"
  echo "  --runtime <runtime>            Container runtime to use (e.g., docker, singularity, apptainer)"
  echo "  --image-dir <path>             Path to store or check for the container image (optional)"
  echo "  --help                         Show this help message and exit"
}

# Parse command-line arguments
benchmark=""
runtime=""
image_dir=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmark)
      benchmark="$2"
      shift 2
      ;;
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
if [[ -z "$benchmark" ]]; then
  echo "Error: --benchmark is required."
  show_usage
  exit 1
fi

# Check if the specified benchmark is valid
if [[ -z "${DEFAULT_IMAGES[$benchmark]}" ]]; then
  echo "Error: Unknown benchmark '$benchmark'."
  show_usage
  exit 1
fi

# Retrieve the image for the specified benchmark
image_name="${DEFAULT_IMAGES[$benchmark]}"

# Check for image locally or in specified directory
if [[ -n "$image_dir" ]]; then
  echo "Checking for image in directory: $image_dir"
  if [[ -f "$image_dir/$(basename $image_name)" ]]; then
    echo "Image found in directory: $image_dir"
    exit 0
  fi
fi

# Download the image
case "$runtime" in
  docker)
    echo "Downloading image with Docker: $image_name"
    docker pull "$image_name" || {
      echo "Error: Failed to download image with Docker.";
      exit 1;
    }
    ;;
  singularity|apptainer)
    echo "Downloading image with $runtime: $image_name"
    $runtime pull docker://$image_name || {
      echo "Error: Failed to download image with $runtime.";
      exit 1;
    }
    ;;
  *)
    echo "Error: Unsupported container runtime '$runtime'."
    show_usage
    exit 1
    ;;
esac

echo "Image downloaded successfully: $image_name"
exit 0

