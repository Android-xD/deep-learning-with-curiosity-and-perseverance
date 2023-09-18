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

git clone git@github.com:qubvel/segmentation_models.pytorch.git
cd segmentation_models.pytorch
/cluster/scratch/$USER/MARS/bin/pip install -e .
cd ..