#!/bin/bash
#SBATCH --job-name=Dec8_2
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16g
#SBATCH -C A100|V100
#SBATCH --output=logs/outputs/Dec8_2_output.txt

module load cuda12.1/toolkit

conda activate dl
python train.py -b 4 -p Dec8_2 -e 50 -d dataset/PhraseCut_1 --msg "Paper hyperparams, large training, corrected dataloader"