#!/bin/bash

# Display usage information
show_usage() {
  echo "Usage: $0 --benchmark <benchmark_name> [--runtime <runtime>] [--image-dir <path>] [--hpl-dat <path>] [other options specific to benchmark]"
  echo ""
  echo "General Options:"
  echo "  --benchmark <benchmark_name>   Name of the benchmark to run (e.g., system_insights, cuda_samples, hpl, osu, nccl, babelstream, imagenet)"
  echo "  --runtime <runtime>            Container runtime to use (e.g., docker, singularity, apptainer)"
  echo "  --image-dir <path>             Path to check for the container image (optional)"
  echo "  --hpl-dat <path>               Path to the HPL.dat file (required for hpl benchmark)"
  echo "  --help                         Show this help message and exit"
  echo ""
  echo "ImageNet Benchmark Options:"
  echo "  --gpus <int>                   Number of GPUs to use (default: 1)"
  echo "  --batch-size <int>             Batch size (default: 256)"
  echo "  --epochs <int>                 Number of epochs (default: 5)"
  echo "  --data-dir <path>              Path to ImageNet or TinyImageNet dataset (default: ./tinyimagenet)"
  echo "  --output-dir <path>            Output log directory (default: ./imagenet_output)"
  echo ""
}

# Parse command-line arguments
benchmark=""
runtime=""
image_dir=""
hpl_dat=""

# Forward all arguments for benchmarks that need more than standard args
forward_args=()

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
  
  bert)
    echo "Launching BERT benchmark..."
    ./benchmarks/bert.sh --runtime "$runtime" "${forward_args[@]}"
    ;;
  *)
      forward_args+=("$1")
      shift
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
  imagenet)
    echo "Launching ImageNet benchmark..."
    ./benchmarks/imagenet.sh --runtime "$runtime" "${forward_args[@]}"
    ;;

  bert)
    echo "Launching BERT benchmark..."
    ./benchmarks/bert.sh --runtime "$runtime" "${forward_args[@]}"
    ;;
  *)
    echo "Error: Unknown benchmark '$benchmark'."
    show_usage
    exit 1
    ;;
esac

exit 0

