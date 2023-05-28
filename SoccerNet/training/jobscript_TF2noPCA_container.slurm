#!/bin/bash
#SBATCH -J CAv1noPCA
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --cpus-per-task=4
#SBATCH --mem=45G
#SBATCH --account=ibex-cs

date
module load singularity

export IMAGE=${BASEDIR}/images/soccernet_0.18_aarch64.sif
export SINGULARITYENV_DATA_DIR=${BASEDIR}/soccerNet/datasets


mkdir -p models/CALF_v1_TF2_noPCA_$SLURM_JOBID
singularity run --nv -B ${BASEDIR},${BASEDIR}/datasets:/data ${IMAGE} python src/main.py \
--SoccerNet_path=/data/SoccerNet/ \
--features=ResNET_TF2.npy  \
--num_features=2048 \
--model_name=CALF_v1_TF2_noPCA \
--version 1 \
--batch_size 1 \
--loss_weight_segmentation 0.02 \
--max_epochs=10 \
--evaluation_frequency=51 \
"$@"
echo "...training done."

date
