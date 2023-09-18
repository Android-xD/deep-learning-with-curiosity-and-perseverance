#!/bin/bash
#SBATCH -n 4
#SBATCH --time=10:00:00
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=4000
#SBATCH --tmp=4000 # per node!!
#SBATCH --job-name=train
#SBATCH --output=../logs/train.out # specify a file to direct output stream
#SBATCH --error=../logs/train.err
#SBATCH --open-mode=truncate # to overrides out and err files, you can also use

source startup.sh
python train.py
