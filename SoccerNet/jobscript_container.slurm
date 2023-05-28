#!/bin/bash
#SBATCH --job-name=SN_feat
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:4
#SBATCH --constraint=a100,4gpus
#SBATCH --reservation=A100
#SBATCH --ntasks=4
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=400G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --account=ibex-cs


module load singularity
export IMAGE=${BASEDIR}/images/soccernet_0.18_aarch64.sif
export SINGULARITYENV_DATA_DIR=${BASEDIR}/soccerNet/datasets


singularity run --nv -B ${BASEDIR} ${IMAGE} mpirun -np 4 ./workload.sh

