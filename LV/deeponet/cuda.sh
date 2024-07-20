#!/bin/bash

# Request 1 CPU core
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -p 3090-gcondo
#SBATCH --gres=gpu:1

#SBATCH -t 24:30:00
#SBATCH -o MyCUDAJob-%j.out
###SBATCH --exclude=gpu1204
###SBATCH --nodelist=gpu2104

# Compile CUDA program and run
#/users/myin2/Environments/pytorch_cuda11/bin

#export SINGULARITY_BINDPATH="/gpfs/scratch,/gpfs/data"
#CONTAINER=/users/myin2/Environments/sing_images/crunch_tf2.simg
#SCRIPT=wgan_gp.py

#singularity exec --nv $CONTAINER python $SCRIPT

#source /users/myin2/Environments/PyTorch-GPU/bin/activate
~/Environments/pytorch_cuda11/bin/python3 main.py --epochs 200000
