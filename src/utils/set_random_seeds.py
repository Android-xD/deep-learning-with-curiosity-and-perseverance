import random
import numpy as np
import torch


def set_seeds(random_seed=0):
    """Set random seeds to get repeatable results."""
    # Set a seed for the random module
    random.seed(random_seed)

    # Set a seed for NumPy
    np.random.seed(random_seed)

    # Set a seed for PyTorch (both CPU and CUDA)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
