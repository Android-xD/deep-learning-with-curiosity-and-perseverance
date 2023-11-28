import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import argparse

import src.utils.configs
from src.utils.configs import trained_weights_dir
from src.data_preprocessing.dataset import ImageDataset, get_transform
from src.models.vae import VAE
from src.models.autoencoder import *
from src.utils.visualize_latent_space import image_scatter, knn, pca, lle, single_cluster, latent_interpolation, mean_shift, tSNE
from src.utils.set_random_seeds import set_seeds
from src.utils import configs
from src.data_preprocessing.dataset import train_test_dataloader

def get_args():
    parser = argparse.ArgumentParser(description='Trainer for autoencoder')
    parser.add_argument('--experiment', default='indices', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    set_seeds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_args()
    experiment = args.experiment
    fig_dir = os.path.join(configs.fig_dir, "ae", experiment)
    os.makedirs(fig_dir, exist_ok=True)
    # Hyperparameters
    batch_size = 8
    input_size = 16
    dataset_name = "perseverance_navcam_color"

    # Data preprocessing and transformation
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop((input_size, input_size)),
        transforms.ToTensor(),
    ])

    # Load the dataset
    dataset = ImageDataset(dataset_name, transform=transform)
    train_dataloader, test_dataloader = train_test_dataloader(dataset, batch_size=batch_size)

    if experiment == "simple":
        ae = AE(3, 16, [8, 16, 32], [8, 16, 32]).to(device)
    elif experiment == "indices":
        ae = Autoencoder().to(device)
        input_size = 32
    else:
        # exeption
        raise NotImplementedError

    state = torch.load(os.path.join(trained_weights_dir, f'autoencoder_{experiment}.pth'), map_location=torch.device(device))
    ae.load_state_dict(state["model_state"])

    # latent_interpolation(ae.encode, ae.decode, dataset, 10, 20,100, filename=os.path.join(fig_dir, "interpolation.png"))
    latent_features = torch.zeros((len(dataset), 16))
    # Loss function and optimizer
    with torch.no_grad():
        for i, (data, _) in enumerate(tqdm(dataset)):
            # move the data to the GPU
            data = data.to(device)
            # Forward pass
            if experiment == "indices":
                latent_features[i] = ae.encoder(data.unsqueeze(0)).ravel()
            else:
                latent_features[i] = ae.encoder(data.unsqueeze(0))[0].ravel()
        # tSNE(latent_features, dataset, filename=os.path.join(fig_dir, "tsne.png"))
        pca(latent_features)
        single_cluster(latent_features, dataset, 30, figure_dir=os.path.join(fig_dir, "kmeans_cluster"))
        mean_shift(latent_features, dataset, figure_dir=os.path.join(fig_dir, "mean_shift_cluster"))
        #knn(latent_features, dataset)
        image_scatter(latent_features[:, 2:], dataset, range(len(dataset)), filename=os.path.join(fig_dir, "latent_space.png"))


