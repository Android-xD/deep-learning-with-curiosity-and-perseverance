"""This module implements the msl dataset with train, validation, and test splits."""

from torch.utils.data import Dataset
from src.utils.configs import data_dir
import numpy as np
from PIL import Image
import os

def load(file):
    """
    returns a tuple of filenames and labels
    """
    table = np.genfromtxt(file, delimiter=" ", dtype=str)
    print(table.shape)
    return table[:, 0], table[:, 1].astype(int)

class MSLDataset(Dataset):
    """
    Mars surface image (Curiosity rover) labeled data set

    This data set consists of 6691 images spanning 24 classes that were collected by the Mars Science Laboratory (MSL,
    Curosity) rover by three instruments (Mastcam Right eye, Mastcam Left eye, and MAHLI). These images are the "browse"
    version of each original data product, not full resolution. They are roughly 256x256 pixels each. We divided the MSL
    images into train, validation, and test data sets according to their sol (Martian day) of acquisition.
    """
    def __init__(self, file, transform=None):
        self.data_dir = data_dir
        self.filenames, self.labels = load(file)
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, "msl_images", self.filenames[idx])
        image = Image.open(img_path, mode="r").convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
