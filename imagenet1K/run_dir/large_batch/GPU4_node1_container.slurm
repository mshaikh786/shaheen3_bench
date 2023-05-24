#!/bin/bash 
#SBATCH --job-name=G4N1B256
#SBATCH --time=0:30:00
#SBATCH --gpus=4
#SBATCH --gpus-per-node=4
#SBATCH --constraint=v100,gpu_ai
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=5
#SBATCH --mem=400G
#SBATCH --output=%x-%J.out
#SBATCH --error=%x-%J.out



module load singularity

export IMAGE=/ibex/ai/home/shaima0d/Shaheen3/Grace_hopper_acceptance/FAT_May_2023/images/torch-horovod_114-0280_amd64.sif

export RUNDIR=${PWD}/result_${SLURM_JOB_NAME}_${SLURM_JOBID}
export PYTHONWARNINGS="ignore"
mkdir -p $RUNDIR

# ImageNet dataset 1000 classes
  ## local storage
if [ -z "${DATA_DIR}" ]; then
  export DATA_DIR="/local/reference/CV/ILSVR/classification-localization/data/jpeg"
fi

batch_size=256
epochs=5
workers=${SLURM_CPUS_PER_TASK}

echo "Hostname: $(/bin/hostname)"
echo "Data source: $DATA_DIR"
echo "Using Batch size : $batch_size"
echo "Epochs : $epochs"
echo "CPU workers: $workers"

BIND_MOUNT='-B /ibex/ai/home/shaima0d/Shaheen3/Grace_hopper_acceptance/FAT_May_2023/ImageNet1K/imagenet1K:/workspace,/ibex/ai/reference/CV:/data'
cd $RUNDIR
main_exe="/workspace/scripts/train_resnet50.py"
#cmd="singularity run --nv ${BIND_MOUNT} $IMAGE python3 ${main_exe} --epochs ${epochs} --batch-size ${batch_size} --num_workers=$workers --root-dir=${DATA_DIR} --train-dir ${DATA_DIR}/train --val-dir ${DATA_DIR}/val ${NODE_LOCAL_STORAGE}"
cmd="singularity run --nv ${BIND_MOUNT} $IMAGE horovodrun -np ${SLURM_NTASKS} python3 ${main_exe} --epochs ${epochs} --batch-size ${batch_size} --num_workers=$workers --root-dir=${DATA_DIR} --train-dir ${DATA_DIR}/train --val-dir ${DATA_DIR}/val ${NODE_LOCAL_STORAGE}"
#time -p srun -u -n ${SLURM_NTASKS} -N ${SLURM_NNODES} -c ${SLURM_CPUS_PER_TASK} ${cmd} --log-dir=log.${SLURM_JOBID} --warmup-epochs=0.0 &> output.txt
time -p ${cmd} --log-dir=log.${SLURM_JOBID} --warmup-epochs=0.0 &> output.txt
    
