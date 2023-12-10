#!/bin/bash
#SBATCH --job-name=Dec9_8
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16g
#SBATCH -C A100|V100
#SBATCH --output=logs/outputs/Dec9_8_output.txt

module load cuda12.1/toolkit

conda activate NST
python train.py -b 4 -p Dec9_8 -e 30 -d dataset/PhraseCut_micro --msg "All 3 allowed- Ablation study"