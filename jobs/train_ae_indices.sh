#!/bin/bash
#SBATCH -n 4
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=4000
#SBATCH --tmp=4000 # per node!!
#SBATCH --job-name=train
#SBATCH --output=./logs/train_ae_indices.out # specify a file to direct output stream
#SBATCH --error=./logs/train_ae_indices.err
#SBATCH --open-mode=truncate # to overrides out and err files, you can also use

source scripts/startup.sh
cd src/training
python train_autoencoder.py --experiment "indices"
cd ../evaluation
python eval_autoencoder.py --experiment "indices"