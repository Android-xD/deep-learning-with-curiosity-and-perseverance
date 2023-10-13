#!/bin/bash
#SBATCH -n 8 # Number of cores
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=4000
#SBATCH --tmp=4000 # per node!!
#SBATCH --job-name=train
#SBATCH --output=./logs/create_dataset.out # specify a file to direct the output stream
#SBATCH --error=./logs/create_dataset.err
#SBATCH --open-mode=truncate

source scripts/startup.sh
cd src/data_preprocessing
python create_dataset.py
