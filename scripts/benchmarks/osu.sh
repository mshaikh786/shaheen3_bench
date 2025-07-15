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
osu_image="library://badawimh/benchmarks/hpcbenchmarks:v1""


workspace_path="/binaries/osu-micro-benchmarks/mpi/pt2pt/"
osu_latency_exec="$workspace_path/osu_latency"
osu_bw_exec="$workspace_path/osu_bw"
osu_bibw_exec="$workspace_path/osu_bibw"

# Check if the image exists in the specified directory
if [[ -n "$image_dir" ]]; then
  echo "Checking for image in directory: $image_dir"
  if [[ ! -f "$image_dir/$(basename $osu_image)" ]]; then
    echo "Error: Image not found in directory: $image_dir"
    exit 1
  fi
fi

# Run the OSU benchmarks
case "$runtime" in
  docker)
    echo "Running OSU benchmarks using Docker..."
    docker run --rm --gpus all -v "$PWD:$workspace_path" $osu_image bash -c "\
      mpirun -np 2 $osu_latency_exec && \
      mpirun -np 2 $osu_bw_exec && \
      mpirun -np 2 $osu_bibw_exec"
    ;;
  singularity|apptainer)
    echo "Running OSU benchmarks using $runtime..."
    $runtime exec --nv $osu_image bash -c "\
      mpirun -np 2 $osu_latency_exec && \
      mpirun -np 2 $osu_bw_exec && \
      mpirun -np 2 $osu_bibw_exec"
    ;;
  *)
    echo "Error: Unsupported container runtime '$runtime'."
    show_usage
    exit 1
    ;;
esac

echo "OSU benchmarks completed successfully."
exit 0

