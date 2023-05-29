#!/bin/bash 
#SBATCH --job-name=G4N1B256
#SBATCH --time=0:30:00
#SBATCH --gpus=4
#SBATCH --gpus-per-node=4
#SBATCH --constraint=a100
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=5
#SBATCH --mem=400G
#SBATCH --output=%x-%J.out
#SBATCH --error=%x-%J.out



export IMAGE=${BASEDIR}/images/torch-horovod-114-0280-v2-aarch64.sif

export RUNDIR=${PWD}/result
export PYTHONWARNINGS="ignore"
mkdir -p $RUNDIR

export DATA_DIR=/workspace/datasets/data/jpeg
mpi_tasks=4
batch_size=256
epochs=5
workers=5

echo "Hostname: $(/bin/hostname)"
echo "Data source: $DATA_DIR"
echo "Using Batch size : $batch_size"
echo "Epochs : $epochs"
echo "CPU workers: $workers"

BIND_MOUNT="-B ${BASEDIR}:/workspace"

cd $RUNDIR
main_exe="/workspace/imagenet1K/scripts/train_resnet50.py"

cmd="singularity run --nv ${BIND_MOUNT} ${IMAGE} horovodrun -np ${mpi_tasks} python3 ${main_exe} --epochs ${epochs} --batch-size ${batch_size} --num_workers=$workers --root-dir=${DATA_DIR} --train-dir ${DATA_DIR}/train --val-dir ${DATA_DIR}/val ${NODE_LOCAL_STORAGE}"

time -p ${cmd} --log-dir=log.txt --warmup-epochs=0.0 &> output.txt

    
