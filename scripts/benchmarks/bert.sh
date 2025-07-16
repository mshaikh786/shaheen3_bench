#!/bin/bash

# Default values
gpus=1
seq_len=128
mBATCH=32
TOTAL_GPUS=1
JOB_NAME="bert_benchmark"
OUTPUT_DIR="./bert_output"
DATA_DIR="/datasets/bert"
CONFIG_DIR="./configs"
SRC_DIR="/workspace/app_benchmarks/bert/src"
SCRIPT_NAME="deepspeed_train.py"
IMAGE="torch_sandbox"
RUNTIME="singularity"
PRINT_STEPS=1
MAX_STEPS=10

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) TOTAL_GPUS="$2"; shift 2 ;;
    --seq-length) seq_len="$2"; shift 2 ;;
    --micro-batch) mBATCH="$2"; shift 2 ;;
    --job-name) JOB_NAME="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --config-dir) CONFIG_DIR="$2"; shift 2 ;;
    --src-dir) SRC_DIR="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --runtime) RUNTIME="$2"; shift 2 ;;
    --help)
      echo "Usage: $0 [options]"
      echo ""
      echo "Options:"
      echo "  --gpus <int>               Total number of GPUs (default: 1)"
      echo "  --seq-length <int>         Sequence length (default: 128)"
      echo "  --micro-batch <int>        Micro-batch size (default: 32)"
      echo "  --job-name <name>          Job name (default: bert_benchmark)"
      echo "  --output-dir <path>        Output directory (default: ./bert_output)"
      echo "  --data-dir <path>          Path to BERT dataset (default: /datasets/bert)"
      echo "  --config-dir <path>        Path to config files (default: ./configs)"
      echo "  --src-dir <path>           Path to training scripts (default: /workspace/app_benchmarks)"
      echo "  --image <image>            Singularity image (default: torch_sandbox)"
      echo ""
      exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

CF_FILE="${CONFIG_DIR}/ksl_bert_large.json"
#mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
abs_outdir=$(readlink -f "$OUTPUT_DIR")

DEEPSPEED_JSON="${CONFIG_DIR}/ksl_bert_large_lamb_b${mBATCH}_seq${seq_len}_GPU_${TOTAL_GPUS}.json"

echo $DEEPSPEED_JSON
export NUM_GPUS=1

echo "Launching DeepSpeed BERT benchmark..."
echo "Output: $OUTPUT_DIR"

export HOSTFILE=hostfile
$RUNTIME run --nv \
  --bind "$(pwd)/app_benchmarks:/workspace/app_benchmarks" $IMAGE  python ${SRC_DIR}/wrapper.py 1

# Bind mount and run benchmark
$RUNTIME run --nv \
  --bind "$(pwd)/app_benchmarks:/workspace/app_benchmarks","$abs_outdir:/output","$CONFIG_DIR:/configs","$DATA_DIR:/data" \
  $IMAGE \
  deepspeed --num_nodes=1 --num_gpus=$TOTAL_GPUS --hostfile ./hostfile \
  $SRC_DIR/$SCRIPT_NAME \
  --cf $CF_FILE \
  --max_seq_length $seq_len \
  --output_dir /output \
  --deepspeed \
  --deepspeed_transformer_kernel \
  --print_steps $PRINT_STEPS \
  --lr_schedule "EE" \
  --lr_offset 10e-4 \
  --job_name $JOB_NAME \
  --deepspeed_config $DEEPSPEED_JSON \
  --data_path_prefix /data \
  --use_nvidia_dataset \
  --max_steps $MAX_STEPS 2>&1 | tee ./output/output.txt

# Run report generation
$RUNTIME run --nv \
  --bind "$(pwd)/app_benchmarks:/workspace/app_benchmarks","$abs_outdir:/output" \
  $IMAGE \
  python3 $SRC_DIR/report.py --file /output/output.txt
