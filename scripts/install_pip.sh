#!/usr/bin/env bash
set -e
source scripts/startup.sh;

mkdir -p /cluster/scratch/$USER/MARS
echo "Creating virtual environment"
python -m venv /cluster/scratch/$USER/MARS
echo "Activating virtual environment"

source /cluster/scratch/$USER/MARS/bin/activate
/cluster/scratch/$USER/MARS/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
/cluster/scratch/$USER/MARS/bin/pip install -r requirements.txt
/cluster/scratch/$USER/MARS/bin/pip install -e .
/cluster/scratch/$USER/MARS/bin/pip install denoising_diffusion_pytorch