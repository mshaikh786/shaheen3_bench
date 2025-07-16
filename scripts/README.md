
## Overview
This project provides an automation framework to execute and manage a variety of GPU-accelerated benchmark tests. The framework uses containerized environments (Docker, Singularity, or Apptainer) to ensure portability and consistency across systems.

## Project Structure
```
scripts/
├── runner.sh          # Main script to orchestrate benchmark execution.
├── download_image.sh  # Script to pull container images for benchmarks.
└── benchmarks/        # Directory containing benchmark-specific scripts.
    ├── system_insights.sh
    ├── hpl.sh
    ├── cuda_samples.sh
    ├── nccl.sh
    ├── babelstream.sh
    ├── osu.sh
    └── imagenet.sh     # ImageNet ResNet50 training benchmark
```

## Benchmarks
The project supports the following benchmarks:

1. **System Insights**:
   - Collects hardware and software system information such as CPU, memory, GPU, and network.

2. **HPL (High-Performance Linpack)**:
   - Runs the HPL benchmark for floating-point performance testing on GPUs.
   - Requires `HPL.dat` configuration file.

3. **CUDA Samples**:
   - Executes basic CUDA utility tests such as device query and bandwidth tests.

4. **NCCL (NVIDIA Collective Communication Library)**:
   - Tests GPU communication performance using NCCL's `all_reduce_perf`.

5. **BabelStream**:
   - Measures memory throughput on GPUs using CUDA streams.

6. **OSU Micro-Benchmarks**:
   - Tests MPI performance metrics, including latency, bandwidth, and bidirectional bandwidth.

7. **ImageNet ResNet50 Application Benchmark**:
   - Trains a ResNet50 model using a TinyImageNet or ImageNet dataset.
   - Uses `train_resnet50.py` inside a container image.
   - Allows configuration of GPU count, batch size, epochs, and dataset directory.
   - Uses the Docker image `mshaikh/ds-torch:270.cu128` and executes via Singularity.

## Prerequisites
- **GPU and Drivers**: Ensure that an NVIDIA GPU is installed and that the appropriate drivers are loaded.
- **Container Runtime**:
  - Docker, Singularity, or Apptainer must be installed.
- **Environment Setup**:
  - Ensure `nvidia-smi` and GPU support via `--nv` in Singularity or Docker.

## Usage

### Run a Benchmark

Use the `runner.sh` script to execute any supported benchmark.

```bash
./runner.sh --benchmark <benchmark_name> --runtime <runtime> [other options]
```

#### Supported Benchmarks:
- `system_insights`
- `hpl`
- `cuda_samples`
- `nccl`
- `babelstream`
- `osu`
- `imagenet`

#### Common Options:
```bash
--runtime <runtime>            Container runtime (e.g., docker, singularity, apptainer)
--image-dir <path>             (Optional) Path to check for the container image
--hpl-dat <path>               Required for hpl benchmark
--help                         Show usage help
```

#### ImageNet Benchmark Options:
```bash
--gpus <int>                   Number of GPUs to use (default: 1)
--batch-size <int>             Batch size (default: 256)
--epochs <int>                 Number of training epochs (default: 5)
--data-dir <path>              Dataset directory (default: ./tinyimagenet)
--output-dir <path>            Output log directory (default: ./imagenet_output)
```

---

## Example Commands

#### Run HPL Benchmark:
```bash
./runner.sh --benchmark hpl --runtime singularity --hpl-dat ./HPL.dat
```

#### Run CUDA Samples Benchmark:
```bash
./runner.sh --benchmark cuda_samples --runtime docker
```

#### Run NCCL Benchmark:
```bash
./runner.sh --benchmark nccl --runtime apptainer
```

#### Run BabelStream Benchmark:
```bash
./runner.sh --benchmark babelstream --runtime singularity
```

#### Run OSU Micro-Benchmarks:
```bash
./runner.sh --benchmark osu --runtime singularity
```

#### Run System Insights:
```bash
./runner.sh --benchmark system_insights --runtime singularity
```

#### Run ImageNet ResNet50 Benchmark (Single GPU):
```bash
./runner.sh --benchmark imagenet --runtime singularity \
  --data-dir /path/to/tinyimagenet \
  --output-dir ./imagenet_output
```

#### Run ImageNet with 4 GPUs, 512 Batch Size, and 10 Epochs:
```bash
./runner.sh --benchmark imagenet --runtime singularity \
  --gpus 4 \
  --batch-size 512 \
  --epochs 10 \
  --data-dir /data/imagenet \
  --output-dir ./results
```

---

## Logs and Outputs
- Each benchmark produces logs or performance metrics.
- Outputs are saved in the current working directory or specified by the `--output-dir` option where applicable.

---

## Extending the Framework

1. **Add a New Benchmark**:
   - Create a new script in the `benchmarks/` directory following the current structure.

2. **Update the Runner Script**:
   - Modify `runner.sh` to include the new benchmark as a case option.
