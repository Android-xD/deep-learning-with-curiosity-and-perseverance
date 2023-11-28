#!/bin/bash
#SBATCH -n 4
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=4000
#SBATCH --tmp=4000 # per node!!
#SBATCH --job-name=train
#SBATCH --output=./logs/colorization-autoencoder.out # specify a file to direct output stream
#SBATCH --error=./logs/colorization-autoencoder.err
#SBATCH --open-mode=truncate # to overrides out and err files, you can also use

source scripts/startup.sh
cd third_party/colorization-autoencoder
python train.py --train_list "perseverance_navcam_color" --parallel 0 --batch-size 32 -j 1 --pth-save-fold "/cluster/scratch/horatan/mars/results" --epochs 100


