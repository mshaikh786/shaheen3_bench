#!/bin/bash

# Display usage information
show_usage() {
  echo "Usage: $0 --benchmark <benchmark_name> [--runtime <runtime>] [--image-dir <path>] [--hpl-dat <path>]"
  echo "Options:"
  echo "  --benchmark <benchmark_name>   Name of the benchmark to run (e.g., system_insights, cuda_samples, hpl, osu, nccl, babelstream)"
  echo "  --runtime <runtime>            Container runtime to use (e.g., docker, singularity, apptainer)"
  echo "  --image-dir <path>             Path to check for the container image (optional)"
  echo "  --hpl-dat <path>               Path to the HPL.dat file (required for hpl benchmark)"
  echo "  --help                         Show this help message and exit"
}

# Parse command-line arguments
benchmark=""
runtime=""
image_dir=""
hpl_dat=""

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
    --hpl-dat)
      hpl_dat="$2"
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

if [[ -z "$runtime" ]]; then
  echo "Error: --runtime is required."
  show_usage
  exit 1
fi

if [[ "$benchmark" == "hpl" && -z "$hpl_dat" ]]; then
  echo "Error: --hpl-dat is required for the hpl benchmark."
  show_usage
  exit 1
fi

# Execute the appropriate benchmark

case "$benchmark" in
  system_insights)
    echo "Launching System Insights benchmark..."
    ./benchmarks/system_insights.sh --runtime "$runtime" --image-dir "$image_dir"
    ;;
  cuda_samples)
    echo "Launching CUDA Samples benchmark..."
    ./benchmarks/cuda_samples.sh --runtime "$runtime" --image-dir "$image_dir"
    ;;
  hpl)
    echo "Launching HPL benchmark..."
    ./benchmarks/hpl.sh --runtime "$runtime" --hpl-dat "$hpl_dat" --image-dir "$image_dir"
    ;;
  osu)
    echo "Launching OSU benchmark..."
    ./benchmarks/osu.sh --runtime "$runtime" --image-dir "$image_dir"
    ;;
  nccl)
    echo "Launching NCCL benchmark..."
    ./benchmarks/nccl.sh --runtime "$runtime" --image-dir "$image_dir"
    ;;
  babelstream)
    echo "Launching BabelStream benchmark..."
    ./benchmarks/babelstream.sh --runtime "$runtime" --image-dir "$image_dir"
    ;;
  *)
    echo "Error: Unknown benchmark '$benchmark'."
    show_usage
    exit 1
    ;;
esac

exit 0

