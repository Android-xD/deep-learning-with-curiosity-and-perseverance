import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from PIL import Image
import cv2
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
from src.utils.color_conversions import lab2rgb_torch, rgb2lab_torch


def load_image(img_path, divisible=32):
    """divisibility by 4 is required for the network"""
    image = Image.open(img_path).convert("RGB")
    width, height = image.size
    # resize to multiple of 4
    transform = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop((height-height%divisible, width-width%divisible)),
        torchvision.transforms.ToTensor(),
    ])

    image = transform(image)

    # convert to lab
    lab = rgb2lab_torch(image)

    # convert to grayscale
    gray = lab[:1, :, :]
    color = lab[1:, :, :]

    return gray, color


if __name__ == '__main__':
    set_seeds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log = []
    fig_dir = os.path.join(configs.fig_dir, "colorizer_curiosity", "images")
    os.makedirs(fig_dir, exist_ok=True)

    # Hyperparameters
    dataset_name = "curiosity_navcam_gray" #"perseverance_navcam_color" # # "curiosity_navcam_gray" # "curiosity_mast_color_small" #

    model = "Reg"
    # perseverance
    # dataset_name = "perseverance_navcam_color"
    #image_list = ["NLF_0719_0730782517_678ECM_N0344394NCAM02719_10_195J01.png", "NLF_0905_0747292317_943ECM_N0442062NCAM03905_04_195J01.png"]
    # curiosity
    dataset_name = "curiosity_navcam_gray"
    image_list = ["NLA_402562693EDR_F0050000NCAM00445M_.JPG", "NLA_401317197EDR_F0042002NCAM00432M_.JPG", "NLA_401751359EDR_F0042100NCAM00307M_.JPG"]
    image_list = [os.path.join(configs.data_dir, "datasets", dataset_name, name) for name in image_list]

    # Initialize model
    if model == "Prob":

        colorizer = Network_Prob().to(device)
    else:
        colorizer = Network6().to(device)
        get_mle = lambda x:x
    checkpoint_file = os.path.join(trained_weights_dir, f'colorizer_{model}.pth')
    state = torch.load(checkpoint_file, map_location=torch.device(device))
    colorizer.load_state_dict(state)

    # Load the dataset

    with torch.no_grad():
        for i, image in enumerate(image_list):
            gray, color = load_image(image)
            gray = gray.unsqueeze(0)
            color = color.unsqueeze(0)
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

            out_img = cv2.cvtColor(256*pred_img[0], cv2.COLOR_RGB2BGR)
            in_img = cv2.cvtColor(256*orig_img[0], cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(fig_dir, f"Colorized_{model}"+ os.path.basename(image_list[i])), out_img)
            cv2.imwrite(os.path.join(fig_dir, os.path.basename(image_list[i])), in_img)
