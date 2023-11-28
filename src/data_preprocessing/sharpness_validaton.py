import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from src.utils.visualization import plot_images, plot_image_pairs
from src.utils.utils import batch2img_list
from src.utils.configs import trained_weights_dir
from src.data_preprocessing.dataset import ImageDataset
from src.models.autoencoder import *
from src.utils.set_random_seeds import set_seeds
from src.utils import configs
from src.data_preprocessing.dataset import train_test_dataloader, get_transform, CIELABDataset
from src.utils.log import epoch_report, plot_losses, plot_losses_batch
from src.utils.utils import spatial_variance

if __name__ == '__main__':
    set_seeds()

    fig_dir = os.path.join(configs.fig_dir, "sharpness")
    os.makedirs(fig_dir, exist_ok=True)

    # Hyperparameters
    input_size = 32
    dataset_name = "perseverance_navcam_color"

    # Data preprocessing and transformation
    transform = get_transform(input_size, True)

    scores = []
    images = []

    # Load the dataset
    dataset = CIELABDataset(dataset_name, transform=transform)
    for i, (data, _) in enumerate(tqdm(dataset)):
        img = batch2img_list(data.unsqueeze(0)*256/100,1)[0]
        images.append(img)
        scores.append(spatial_variance(img))
        if i == 1000:
            break
    order = np.argsort(scores)
    i = 0
    img_list = []
    ts = []
    for t in [0,1,2,20,50,100,150, 200]:

        while i < len(order) and scores[order[i]] < t:
            i+=1

        if i == len(order):
            break
        img_list.append(images[order[i]])
        ts.append(f'{scores[order[i]]:.2f}')
        i += 1
    plot_images(img_list, ts, hpad=1.5)
    plt.show()