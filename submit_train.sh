#!/bin/bash
#SBATCH --gpus-per-node=v100l:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=23:59:00
#SBATCH --account=def-timsbc
#SBATCH --job-name=DFormer-3DCC
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=yuxinchen@ece.ubc.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
set -e

# Activate the existing virtual environment
source ~/projects/def-timsbc/ychen506/DFormer-3DCC/dformer/bin/activate

# Run the training script
bash train.sh