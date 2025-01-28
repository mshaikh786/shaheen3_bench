## Overview
This project provides an automation framework to execute and manage a variety of GPU-accelerated benchmark tests. The framework uses containerized environments (Docker, Singularity, or Apptainer) to ensure portability and consistency across systems.

## Project Structure
```
project-root/
├── runner.sh          # Main script to orchestrate benchmark execution.
├── download_image.sh  # Script to pull container images for benchmarks.
└── benchmarks/        # Directory containing benchmark-specific scripts.
    ├── system_insights.sh
    ├── hpl.sh
    ├── cuda_samples.sh
    ├── nccl.sh
    ├── babelstream.sh
    └── osu.sh
```

## Benchmarks
The project supports the following benchmarks:

1. **System Insights** (`system_insights.sh`):
   - Collects hardware and software system information such as CPU, memory, GPU, and network.

2. **HPL (High-Performance Linpack)** (`hpl.sh`):
   - Runs the HPL benchmark for floating-point performance testing on GPUs.
   - Requires `HPL.dat` configuration.

3. **CUDA Samples** (`cuda_samples.sh`):
   - Executes basic CUDA utility tests such as device query and bandwidth tests.

4. **NCCL (NVIDIA Collective Communication Library)** (`nccl.sh`):
   - Tests GPU communication performance using NCCL's `all_reduce_perf`.

5. **BabelStream** (`babelstream.sh`):
   - Measures memory throughput on GPUs using CUDA streams.

6. **OSU Micro-Benchmarks** (`osu.sh`):
   - Tests MPI performance metrics, including latency, bandwidth, and bidirectional bandwidth.

## Prerequisites
- **GPU and Drivers**: Ensure that an NVIDIA GPU is installed and that the appropriate drivers are loaded.
- **Container Runtime**:
  - Docker, Singularity, or Apptainer must be installed.
- **Environment Setup**:
  - The `nvcc` and `nvidia-smi` commands should be available.

## Usage

### 1. Run a Benchmark
Use the `runner.sh` script to execute any supported benchmark. The script accepts the following options:

```bash
Usage: ./runner.sh --benchmark <benchmark_name> --runtime <runtime> [options]

Options:
  --benchmark <benchmark_name>   Name of the benchmark to run (e.g., system_insights, hpl, cuda_samples, nccl, babelstream, osu).
  --runtime <runtime>            Container runtime to use (e.g., docker, singularity, apptainer).
  --image-dir <path>             (Optional) Path to check for container images.
  --hpl-dat <path>               (Required for HPL) Path to the HPL.dat configuration file.
  --help                         Show usage information.
```

### 2. Download Required Images
The `download_image.sh` script documents the process of manually pulling container images for a specific benchmark. However, note that the required image will be automatically downloaded by the `runner.sh` script if it does not already exist.

```bash
Usage: ./download_image.sh --benchmark <benchmark_name> --runtime <runtime> [--image-dir <path>]
```

### Example Commands
#### Run the HPL Benchmark:
```bash
./runner.sh --benchmark hpl --runtime singularity --hpl-dat ./HPL.dat
```

#### Run the CUDA Samples Benchmark:
```bash
./runner.sh --benchmark cuda_samples --runtime docker
```

#### Document the NCCL Benchmark Image Download:
```bash
./download_image.sh --benchmark nccl --runtime docker
```

## Benchmark Scripts

Each benchmark script in the `benchmarks/` directory can also be run independently. Below is a brief overview of their usage:

### `system_insights.sh`
```bash
Usage: ./benchmarks/system_insights.sh --runtime <runtime> [--image-dir <path>]
```

### `hpl.sh`
```bash
Usage: ./benchmarks/hpl.sh --runtime <runtime> --hpl-dat <path> [--image-dir <path>]
```

### `cuda_samples.sh`
```bash
Usage: ./benchmarks/cuda_samples.sh --runtime <runtime> [--image-dir <path>]
```

### `nccl.sh`
```bash
Usage: ./benchmarks/nccl.sh --runtime <runtime> [--image-dir <path>]
```

### `babelstream.sh`
```bash
Usage: ./benchmarks/babelstream.sh --runtime <runtime> [--image-dir <path>]
```

### `osu.sh`
```bash
Usage: ./benchmarks/osu.sh --runtime <runtime> [--image-dir <path>]
```

## Logs and Outputs
- Each benchmark produces logs or performance metrics.
- Outputs are saved in the current working directory or specified by the `--output` option where applicable.

## Extending the Framework
1. **Add a New Benchmark**:
   - Create a new script in the `benchmarks/` directory.
   - Follow the structure of the existing scripts for consistency.

2. **Modify Runner Script**:
   - Add support for the new benchmark in `runner.sh`.


