import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import json

import src.utils.configs
from src.utils.visualization import plot_images
from src.utils.utils import batch2img_list
from src.utils.set_random_seeds import set_seeds
from src.utils.configs import trained_weights_dir
from src.data_preprocessing.dataset import CIELABDataset, ImageDataset
from src.utils import configs
from src.models.colorizer import Network6
from src.utils.color_conversions import lab2rgb_torch


if __name__ == '__main__':
    set_seeds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log = []
    fig_dir = os.path.join(configs.fig_dir, "colorizer")
    os.makedirs(fig_dir, exist_ok=True)

    # Hyperparameters
    batch_size = 8
    num_epochs = 10
    input_size = 128
    dataset_name ="curiosity_navcam_gray"
    #"perseverance_mast_color"

    # Initialize model
    colorizer = Network6().to(device)
    checkpoint_file = os.path.join(trained_weights_dir, 'colorizer.pth')
    state = torch.load(checkpoint_file, map_location=torch.device(device))
    colorizer.load_state_dict(state)

    # Data preprocessing and transformation
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomCrop((input_size, input_size)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    # Load the dataset
    dataset = CIELABDataset(dataset_name, transform=transform)
    src.utils.visualization.plot_dataset(dataset, 5, 7)
    plt.show()

    val_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    with torch.no_grad():
        for i, (gray, color) in enumerate(tqdm(val_dataloader)):
            # move the data to the GPU
            gray, color = gray.to(device), color.to(device)
            # Forward pass
            pred = colorizer(gray)
            pred_img = torch.cat((gray, pred.detach()), dim=1)
            pred_img = lab2rgb_torch(pred_img)
            pred_img = batch2img_list(pred_img, n_max=8)
            plot_images(pred_img,)
            plt.show()
