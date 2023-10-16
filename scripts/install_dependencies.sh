#!/usr/bin/env bash

# this script must be called from the repository root
mkdir third_party
cd third_party

# install segmentation models
git clone git@github.com:qubvel/segmentation_models.pytorch.git
cd segmentation_models.pytorch
/cluster/scratch/$USER/MARS/bin/pip install -e . --no-dependencies
cd ..

# install dino repo from my fork
git clone git@github.com:Android-xD/dino.git
cd dino
/cluster/scratch/$USER/MARS/bin/pip install -e . --no-dependencies
cd ..

# install diffusion with pip
# git clone https://github.com/lucidrains/denoising-diffusion-pytorch.git
/cluster/scratch/$USER/MARS/bin/pip install denoising_diffusion_pytorch

# install dinov2 from github
git clone https://github.com/facebookresearch/dinov2.git
cd dinov2
/cluster/scratch/$USER/MARS/bin/pip install -e . --no-dependencies
cd ..

# go back to repository root
cd ..