#!/bin/bash 
#SBATCH --job-name=G4N1B256
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH -p amp
#SBATCH --constraint=gpu,a100


nvidia-smi 
nvidia-smi topo -m 

export BASEDIR=/global/scratch/users/hateml/mohsin/shaheen3_bench
source ~/mohsin/activate_apptainer.sh
export IMAGE=${BASEDIR}/images/torch-horovod-114-0280-v2-aarch64.sif 
export rundir=${PWD}
export base_dir=${rundir}/..
export config_files=${base_dir}/config_files
export src=${base_dir}/src
export SINGULARITYENV_PREPEND_PATH=/software/bin
export SINGULARITYENV_PYTHONPATH=/software/lib/python3.8/site-packages

export DATA_DIR=/data

export NUM_GPUS=4 #$(echo $SLURM_JOB_GPUS | awk -F "," '{print NF}')
export TOTAL_GPUS=4 #$(expr ${NUM_GPUS} \* ${SLURM_NNODES})



SEQ_LEN=128
mBATCH=64
JOB_NAME=lamb_nvidia_data_${mBATCH}_seq${SEQ_LEN}_${TOTAL_GPUS}_GPU.${SLURM_JOBID}
OUTPUT_DIR=${rundir}/model_${TOTAL_GPUS}_GPU_${SLURM_JOBID}
mkdir -p $OUTPUT_DIR


#export NCCL_TREE_THRESHOLD=0 
#export NCCL_SOCKET_IFNAME=ib0

export BIND_MOUNT="-B ${BASEDIR},${BASEDIR}/datasets/bert_dataset:/data"

export HOSTFILE=hostfile
singularity run ${BIND_MOUNT} ${IMAGE} python ${src}/wrapper.py 1


time -p singularity run --nv ${BIND_MOUNT} ${IMAGE} deepspeed --num_nodes=1 --num_gpus=${TOTAL_GPUS} --hostfile ./hostfile  ${src}/deepspeed_train.py \
--cf ${config_files}/ksl_bert_large.json \
--max_seq_length ${SEQ_LEN} \
--output_dir $OUTPUT_DIR \
--deepspeed \
--deepspeed_transformer_kernel \
--print_steps 1 \
--lr_schedule "EE" \
--lr_offset 10e-4 \
--job_name $JOB_NAME \
--deepspeed_config ${config_files}/ksl_bert_large_lamb_b${mBATCH}_seq${SEQ_LEN}_GPU_${TOTAL_GPUS}.json \
--data_path_prefix ${DATA_DIR} \
--use_nvidia_dataset \
--max_steps 10

