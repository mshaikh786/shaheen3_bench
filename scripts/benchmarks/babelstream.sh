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
babelstream_image="nvcr.io/nvidia/babelstream:latest"
workspace_path="/workspace"
cuda_stream_exec="$workspace_path/cuda-stream"
arraysize=$(( 2040*1024*1024 ))
stdout="STDOUT.txt"

# Check if the image exists in the specified directory
if [[ -n "$image_dir" ]]; then
  echo "Checking for image in directory: $image_dir"
  if [[ ! -f "$image_dir/$(basename $babelstream_image)" ]]; then
    echo "Error: Image not found in directory: $image_dir"
    exit 1
  fi
fi

# Run the BabelStream benchmark
case "$runtime" in
  docker)
    echo "Running BabelStream benchmark using Docker..."
    docker run --rm --gpus all -v "$PWD:$workspace_path" $babelstream_image bash -c "\
      echo -e '\nUsing device: 0\n' | tee -a $stdout && \
      $cuda_stream_exec --device 0 --arraysize $arraysize | tee -a $stdout && \
      awk '/Triad/ {T=T+\$2} END{printf \"\nTriad(node) %11.3f MB/s\n\",T}' $stdout | tee -a $stdout"
    ;;
  singularity|apptainer)
    echo "Running BabelStream benchmark using $runtime..."
    $runtime exec --nv docker://$babelstream_image bash -c "\
      echo -e '\nUsing device: 0\n' | tee -a $stdout && \
      $cuda_stream_exec --device 0 --arraysize $arraysize | tee -a $stdout && \
      awk '/Triad/ {T=T+\$2} END{printf \"\nTriad(node) %11.3f MB/s\n\",T}' $stdout | tee -a $stdout"
    ;;
  *)
    echo "Error: Unsupported container runtime '$runtime'."
    show_usage
    exit 1
    ;;
esac

echo "BabelStream benchmark completed successfully."
exit 0

