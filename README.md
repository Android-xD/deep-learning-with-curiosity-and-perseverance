# deep-learning-with-curiosity-and-perseverance

To maintain a clean, organized, and reproducible codebase, we follow a structured layout. Below is an overview of the repository structure:

```
deep-learning-with-curiosity-and-perseverance/
│
├── data/
│   ├── curiosity/
│   │   ├── CHEMCAM/
│   │   │   ├── CR0_397506222EDR_F0010008CCAM00000M_.JPG
│   │   │   └── ...
│   │   └── .../
│   ├── perceverance/
│   │   ├── NAVCAM_LEFT/
│   │   │   ├── NLB_0906_0747368827_599ECM_N0442062NCAM00502_03_1I6J02.png
│   │   │   └── ...
│   │   └── .../
│   └── urls/
│        ├── curiosity/
│        │   ├── 0000.json
│        │   ├── ...
│        │   └── 3944.json
│        └── perceverance/
│            ├── 0000.json
│            ├── ...
│            └── 0558.json
├── notebooks/
│
├── src/
│   ├── data_preprocessing/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── utils/
│
├── experiments/
│   ├── experiment_1/
│   ├── experiment_2/
│   └── ...
│
├── configs/
│
├── requirements.txt
│
├── README.md
│
└── main.py

```

**Directory Structure Explanation:**

1. **data/**: This directory is reserved for data management.
   - **curiosity/**: Store the original, unprocessed data here.
   - **perceverance/**: Store the original, unprocessed data here.
   - **datasets/**: If custom datasets are created, they should reside here.

2. **notebooks/**: This is where you can find Jupyter notebooks used for data exploration and experimentation.

3. **src/**: The source code is organized into subdirectories:
   - **data_preprocessing/**: Contains code for data loading, augmentation, and transformation.
   - **models/**: Define your neural network architectures here.
   - **training/**: Includes code for training loops, loss functions, and optimizers.
   - **evaluation/**: Houses code for model evaluation, metric calculation, etc.
   - **utils/**: Utility functions and helper scripts that can be reused.

4. **experiments/**: Each experiment has its own subdirectory, containing experiment-specific code, configuration files, and logs.

5. **configs/**: Configuration files (e.g., JSON or YAML) are stored here. These files define hyperparameters, model architectures, and other settings, facilitating experiment reproducibility.

6. **requirements.txt**: List the project's dependencies and their versions, ensuring others can recreate your environment.

7. **README.md**: You are currently reading this file! It serves as documentation for the project, offering an overview, installation instructions, usage examples, and relevant information.

8. **main.py**: This main script orchestrates the overall project workflow. It can parse command-line arguments and invoke functions from `src/` modules to train models, evaluate results, and more.

# Installation on Euler cluster
```commandline
cd scripts
sbatch job_install_pip.sh
source startup.sh
cd ..
cd segmentation_models.pytorch
pip install -e .
```
or 
```commandline
cd scripts
source install_pip.sh
```
All the bash scripts need to be called from the scripts folder otherwise the paths break.