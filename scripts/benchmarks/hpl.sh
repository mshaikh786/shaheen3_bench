#!/bin/bash

# Display usage information
show_usage() {
  echo "Usage: $0 --runtime <runtime> --hpl-dat <path> [--image-dir <path>] [--cpu-affinity <range>] [--output <path>]"
  echo "Options:"
  echo "  --runtime <runtime>            Container runtime to use (e.g., docker, singularity, apptainer)"
  echo "  --hpl-dat <path>               Path to the HPL.dat configuration file"
  echo "  --image-dir <path>             Path to check for the container image (optional)"
  echo "  --cpu-affinity <range>         CPU affinity range to bind processes (e.g., 0-71)"
  echo "  --output <path>                Path for storing the benchmark output (default: ./stdout.txt)"
  echo "  --help                         Show this help message and exit"
}

# Parse command-line arguments
runtime=""
hpl_dat=""
image_dir=""
cpu_affinity="0-15"  # Default CPU affinity
output="./stdout.txt"  # Default output file

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
    --cpu-affinity)
      cpu_affinity="$2"
      shift 2
      ;;
    --output)
      output="$2"
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

# Define image
hpc_bench_image="nvcr.io/nvidia/hpc-benchmarks:24.09"

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
      mpirun -np 1 -cpus-per-proc 16 hpl.sh --dat $hpl_dat --cpu-affinity $cpu_affinity > $output"
    ;;
  singularity|apptainer)
    echo "Running HPL benchmark using $runtime..."
    $runtime exec --nv --bind "$(dirname $hpl_dat):/dat-files" docker://$hpc_bench_image bash -c "\
      mpirun -np 1 -cpus-per-proc 16 /workspace/hpl.sh --dat /dat-files/$(basename $hpl_dat) --cpu-affinity $cpu_affinity > $(basename $output)"
    ;;
  *)
    echo "Error: Unsupported container runtime '$runtime'."
    show_usage
    exit 1
    ;;
esac

echo "HPL benchmark completed successfully. Output saved to $output."
exit 0

