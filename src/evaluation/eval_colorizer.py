import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import json

import src.utils.configs
from src.utils.visualization import plot_images, plot_image_pairs
from src.utils.utils import batch2img_list
from src.utils.set_random_seeds import set_seeds
from src.utils.configs import trained_weights_dir
from src.data_preprocessing.dataset import CIELABDataset, ImageDataset, get_transform
from src.utils import configs
from src.models.colorizer import Network6, Network_Prob, get_mle
from src.utils.color_conversions import lab2rgb_torch

def pixel_scatter(image, filename=None):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
    """ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)"""

    plt.scatter(image[:, :, 0].flatten(), image[:, :, 1].flatten(), image[:, :, 2].flatten(), c=image.reshape(-1, 3))
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=600, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def color_histogram(image, title=None, filename=None):
    fig = plt.figure()
    if title:
        fig.suptitle(title)
    plt.hist(image[:,:,0].flatten(), color='red', alpha=0.5, bins=100)
    # green channel
    plt.hist(image[:,:,1].flatten(), color='green', alpha=0.5, bins=100)
    # blue channel
    plt.hist(image[:,:,2].flatten(), color='blue', alpha=0.5, bins=100)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=600, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def pixel_scatter_2d(gray, color, title=None, filename=None):
    """
    Three subplots, that show the correlation between the gray color,
    and each of the RGB channels.

    The y-axis is the gray value, the x-axis is the RGB value.

    """
    img = torch.cat((gray, color), dim=1)
    img = lab2rgb_torch(img)
    img = batch2img_list(img, n_max=8)
    fig = plt.figure()

    if title:
        fig.suptitle(title)
    for i, C in enumerate(['R', 'G', 'B']):
        ax = fig.add_subplot(1, 3, i+1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        if i == 0:
            ax.set_ylabel('gray')
        ax.set_xlabel(C)
        plt.scatter(img[0][:, :, i].flatten(), gray[0, 0, :, :].flatten()/100, c=img[0].reshape(-1, 3), s=3, alpha=0.5)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=600, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def pixel_scatter_lab(gray, color, title=None, filename=None):
    """
    Three subplots, that show the correlation between the gray color,
    and each of the RGB channels.

    The y-axis is the gray value, the x-axis is the RGB value.

    """

    img = torch.cat((gray, color), dim=1)
    img = lab2rgb_torch(img)
    img = batch2img_list(img, n_max=8)
    fig = plt.figure()

    if title:
        fig.suptitle(title)
    # create one scatter plot
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(-10, 4)
    ax.set_ylim(5, 40)
    ax.set_xlabel('A')
    ax.set_ylabel('B')


    ax.scatter(color[0, 0, :, :].flatten(), color[0, 1, :, :].flatten(), c=img[0].reshape(-1, 3), s=15, alpha=0.5)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=600, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    set_seeds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log = []
    experiment = "Reg"
    fig_dir = os.path.join(configs.fig_dir, f"colorizer_{experiment}","eval")
    os.makedirs(fig_dir, exist_ok=True)

    # Hyperparameters
    batch_size = 8
    input_size = 64
    dataset_name = "perseverance_navcam_color" # "curiosity_navcam_gray" # "curiosity_navcam_gray" # "curiosity_mast_color_small" #
        # "curiosity_navcam_gray"
    #

    # Initialize model
    if experiment == "Reg":
        get_mle = lambda x:x
        colorizer = Network6().to(device)
    else:
        colorizer = Network_Prob().to(device)

    checkpoint_file = os.path.join(trained_weights_dir, f'colorizer_{experiment}.pth')
    state = torch.load(checkpoint_file, map_location=torch.device(device))
    colorizer.load_state_dict(state)

    # Data preprocessing and transformation
    transform = get_transform(input_size,augment=True)

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
            pred = get_mle(pred)
            pred_img = torch.cat((gray, pred.detach()), dim=1)
            pred_img = lab2rgb_torch(pred_img)
            pred_img = batch2img_list(pred_img, n_max=8)

            orig_img = torch.cat((gray, color), dim=1)
            orig_img = lab2rgb_torch(orig_img)
            orig_img = batch2img_list(orig_img, n_max=8)

            plot_image_pairs(orig_img, pred_img,filename=os.path.join(fig_dir, f"colorizer_{i}.png"))

            plot_images(pred_img)
            plt.show()
            plot_images(orig_img)
            plt.show()

            # color histogram
            color_histogram(pred_img[0], title="predicted")
            plt.show()
            color_histogram(orig_img[0], title="original")
            plt.show()

            # 2d scatter plot of all pixels, with color as rgb
            pixel_scatter_2d(gray[:1], color[:1], "original", filename=os.path.join(fig_dir, f"colorizer_2dscatter_{i}_original.png"))
            pixel_scatter_2d(gray[:1], pred.detach()[:1], "predicted", filename=os.path.join(fig_dir, f"colorizer_2dscatter_{i}_predicted.png"))

            # 2d scatter plot of all pixels, with color as rgb
            pixel_scatter_lab(gray[:1], color[:1], "original", filename=os.path.join(fig_dir, f"colorizer_labscatter_{i}_original.png"))
            pixel_scatter_lab(gray[:1], pred.detach()[:1], "predicted", filename=os.path.join(fig_dir, f"colorizer_labscatter_{i}_predicted.png"))

            # 3d scatter plot of all pixels, with color as rgb
            #pixel_scatter(pred_img[0], filename=os.path.join(fig_dir, f"colorizer_3dscatter_{i}.png"))
            #pixel_scatter(orig_img[0], filename=os.path.join(fig_dir, f"colorizer_3dscatter_{i}.png"))