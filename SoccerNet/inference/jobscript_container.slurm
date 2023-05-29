#!/bin/bash 
#SBATCH --job-name=G4N1
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH -p amp
#SBATCH --constraint=gpu,a100


nvidia-smi 
nvidia-smi topo -m 

# IF using local installed Apptainer, please update the PATH
# export PATH=/apptainer_install_dir/aarch/bin:$PATH


export IMAGE=${BASEDIR}/images/soccernet_0.18_aarch64.sif
export SINGULARITYENV_DATA_DIR=/data
export BIND_MOUNT="-B ${BASEDIR},${BASEDIR}/datasets/soccernet_dataset:/data"

singularity run --nv ${BIND_MOUNT} ${IMAGE} mpirun -np 4 ./workload.sh

