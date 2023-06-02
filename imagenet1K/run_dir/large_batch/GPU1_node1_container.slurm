#!/bin/bash 
#SBATCH --job-name=G4N1B256
#SBATCH --time=0:30:00
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=400G
#SBATCH --output=%x-%J.out
#SBATCH --error=%x-%J.out


nvidia-smi
nvidia-smi top -m 

export IMAGE=${BASEDIR}/images/torch-horovod-114-0280-v2-aarch64.sif

export RUNDIR=${PWD}/result
export PYTHONWARNINGS="ignore"
mkdir -p $RUNDIR

export DATA_DIR=/workspace/datasets/data/jpeg

mpi_tasks=1
batch_size=256
epochs=5
workers=5

echo "Hostname: $(/bin/hostname)"
echo "Data source: $DATA_DIR"
echo "Using Batch size : $batch_size"
echo "Epochs : $epochs"
echo "CPU workers: $workers"

BIND_MOUNT="-B ${BASEDIR}:/workspace,/ibex/ai/reference/CV/tinyimagenet"

cd $RUNDIR
main_exe="/workspace/imagenet1K/scripts/train_resnet50.py"

cmd="singularity run --nv ${BIND_MOUNT} ${IMAGE} horovodrun -np ${mpi_tasks} python3 ${main_exe} --epochs ${epochs} --batch-size ${batch_size} --num_workers=$workers --root-dir=${DATA_DIR} --train-dir ${DATA_DIR}/train --val-dir ${DATA_DIR}/val ${NODE_LOCAL_STORAGE}"

time -p ${cmd} --log-dir=log.txt --warmup-epochs=0.0 &> output.txt

    
