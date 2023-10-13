import glob
import os
from PIL import Image

from torch.utils.data import Dataset
from src.utils.color_conversions import rgb2lab_torch
from src.utils.configs import data_dir

class ImageDataset(Dataset):
    """
        ImageDataset class for loading images from a file pattern
        transform: torchvision.transforms
    """
    def __init__(self, dataset_name, transform=None):
        self.image_files = glob.glob(os.path.join(data_dir, dataset_name, "*"))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image

class GrayDataset(Dataset):
    def __init__(self, dataset_name, transform=None):
        self.image_files = glob.glob(os.path.join(data_dir, dataset_name, "*"))

        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # convert to lab
        lab = rgb2lab_torch(image)

        # convert to grayscale
        gray = lab[:1, :, :]
        color = lab[1:, :, :]

        return gray, color
