#!/bin/bash

# Default values
gpus=1
batch_size=256
epochs=5
data_dir="./tinyimagenet"
output_dir="./imagenet_output"
runtime="singularity"
docker_image="docker://mshaikh/ds-torch:270.cu128"
train_script="/workspace/app_benchmarks/train_resnet50.py"

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) gpus="$2"; shift 2 ;;
    --batch-size) batch_size="$2"; shift 2 ;;
    --epochs) epochs="$2"; shift 2 ;;
    --data-dir) data_dir="$2"; shift 2 ;;
    --output-dir) output_dir="$2"; shift 2 ;;
    --runtime) runtime="$2"; shift 2 ;;
    --help)
      echo "Usage: $0 --gpus <n> --batch-size <bs> --epochs <e> --data-dir <path> --output-dir <path>"
      exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

mkdir -p "$output_dir"

echo "Running ImageNet training inside $docker_image using $runtime..."

singularity exec --nv \
  --bind "$data_dir:/data","$output_dir:/out","$(pwd)/app_benchmarks:/workspace/app_benchmarks" \
  $docker_image \
    bash -c "export PYTHONPATH=/software/local/lib/python3.12/dist-packages/horovod-0.28.0-py3.12-linux-x86_64.egg:\$PYTHONPATH && \
     python3 $train_script \
    --epochs "$epochs" \
    --batch-size "$batch_size" \
    --num_workers 4 \
    --root-dir /data \
    --train-dir /data/train \
    --val-dir /data/val \
    --log-dir /out \
    --warmup-epochs 0.0 "

