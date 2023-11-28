#!/bin/bash
#SBATCH -n 4
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=4000
#SBATCH --tmp=4000 # per node!!
#SBATCH --job-name=train
#SBATCH --output=./logs/cluster_imagenet-autoencoder.out # specify a file to direct output stream
#SBATCH --error=./logs/cluster_imagenet-autoencoder.err
#SBATCH --open-mode=truncate # to overrides out and err files, you can also use

source scripts/startup.sh
cd third_party/imagenet-autoencoder
python visualize_clustering.py --arch "vgg16" \
 --val_list "perseverance_navcam_color" \
 --resume "/cluster/scratch/horatan/mars/results_vgg16/051.pth"