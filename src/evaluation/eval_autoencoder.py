import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os

import src.utils.configs
from src.utils.configs import trained_weights_dir
from src.data_preprocessing.dataset import ImageDataset
from src.models.vae import VAE
from src.utils.visualize_latent_space import image_scatter, knn, pca, lle,single_cluster
from src.utils.set_random_seeds import set_seeds
from src.utils import configs
from src.data_preprocessing.dataset import train_test_dataloader

if __name__ == '__main__':
    set_seeds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fig_dir = os.path.join(configs.fig_dir, "vae")
    os.makedirs(fig_dir, exist_ok=True)
    # Hyperparameters
    batch_size = 8


    # Data preprocessing and transformation
    transform = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.ToTensor(),
    ])

    # Load the dataset
    dataset = ImageDataset("perseverance_navcam_color", transform=transform)
    train_dataloader, test_dataloader = train_test_dataloader(dataset, batch_size=batch_size)

    # Initialize the encoder and decoder
    variational_encoder = VAE()
    state = torch.load(os.path.join(trained_weights_dir, 'vae.pth'), map_location=torch.device(device))
    variational_encoder.load_state_dict(state['model_state'])

    latent_features = torch.zeros((len(dataset), 32))
    # Loss function and optimizer
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataset)):
            # move the data to the GPU
            data = data.to(device)
            # Forward pass
            latent_features[i] = variational_encoder.encode(data.unsqueeze(0)).ravel()
        #latent_features -= torch.mean(latent_features, dim=0)
        #latent_features /= torch.std(latent_features, dim=0)
        pca(latent_features)
        single_cluster(latent_features, dataset,30, figure_dir=fig_dir)
        #knn(latent_features, dataset)
        image_scatter(latent_features[:,2:], dataset, range(len(dataset)))


