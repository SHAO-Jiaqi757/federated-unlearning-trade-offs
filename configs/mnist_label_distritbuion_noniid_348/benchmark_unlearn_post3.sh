#!/bin/bash

#SBATCH --job-name=FU_baseline   # Job name
#SBATCH --output=logs/FU_%A_%a.log # Output and error log (%A is the array job ID, %a is the array task ID)
#SBATCH --nodes=1                        # Number of nodes per task
#SBATCH --ntasks-per-node=2               # Number of tasks per node
#SBATCH --cpus-per-task=8                 # Number of CPU cores per task
#SBATCH --partition=gpu-share             # Partition to submit to
#SBATCH --time=0-12:00:00                 # Time limit hrs:min:sec or days-hours:minutes:seconds
#SBATCH --gres=gpu:2                     # GPUs per task

# Load modules or source your Python environment here if needed
# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate openmmlab

# Run the Python script with arguments
~/miniconda3/envs/openmmlab/bin/python ./main.py -c /home/jshaoaj/project/unlearning/benchmark_FU/configs/mnist_label_distritbuion_noniid_348/config_unlearn_post3.yml --resume
