#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --job-name=lit_cml

#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --account=scavenger

#SBATCH --nodes=1           # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:rtxa4000:8
#SBATCH --ntasks-per-node=8   # This needs to match Trainer(devices=...)
#SBATCH --mem=0
#SBATCH --time=24:00:00

#SBATCH --output=log/%x_%A_%a.log
#SBATCH --error=log/%x_%A_%a.log
#SBATCH --open-mode=append

# debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo

# might need the latest CUDA
# module load NCCL/2.4.7-1-cuda.10.0


echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

srun python train.py \
    --fabric_accelerator cuda \
    --fabric_strategy ddp \
    --fabric_devices $SLURM_NTASKS_PER_NODE \
    --fabric_num_nodes $SLURM_JOB_NUM_NODES \
    --fabric_precision 'bf16-mixed' \
    --batch_size 2
