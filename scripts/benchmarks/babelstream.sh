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

# Define image and paths
workspace_path="./benchmarks/binaries"
cuda_stream_exec="$workspace_path/cuda-stream"
arraysize=$(( 2040*1024*1024 ))
stdout="STDOUT.txt"

# Run the BabelStream benchmark

echo -e '\nUsing device: 0\n' | tee -a $stdout && \
$cuda_stream_exec --device 0 --arraysize $arraysize | tee -a $stdout && \
awk '/Triad/ {T=T+$2} END{printf "\nTriad(node) %11.3f MB/s\n",T}' ${stdout}


echo "BabelStream benchmark completed successfully."
exit 0

