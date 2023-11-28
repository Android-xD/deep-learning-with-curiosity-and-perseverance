import os
import glob

import cv2
import torch
import seaborn as sns
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import src.utils.configs as configs
from src.utils.utils import human_readable_size
from src.utils.visualization import plot_images, image_scatter_plot, plot_image_grid
from src.utils.utils import spatial_variance, shannon_entropy, crop_black_border


def create_image_samples(rover="curiosity"):
    """Plot a few image samples for each camera type. """
    rover_dir = os.path.join(configs.data_dir, rover)
    camera_types = os.listdir(rover_dir)
    camera_types = [name.replace("LEFT", "*") for name in camera_types]
    camera_types = [name.replace("RIGHT", "*") for name in camera_types]
    camera_types = set(camera_types)

    img_lists = []

    for i, cam_type in enumerate(camera_types):
        img_list = glob.glob(os.path.join(rover_dir, cam_type, "*"))
        samples = np.arange(len(img_list))
        np.random.shuffle(samples)
        imgs = []
        for j in samples:
            img = cv2.imread(img_list[j])
            if not img is None:
                imgs.append(img[:, :, ::-1])
            if len(imgs) > 6:
                break

        plot_images(imgs)
        path = os.path.join(configs.fig_dir, rover, cam_type + ".png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        path = path.replace("_*", "")
        plt.savefig(path)

        img_lists += [imgs]
    plot_image_grid(img_lists, list(camera_types))
    path = os.path.join(configs.fig_dir, rover, "GRID.png")
    plt.savefig(path)


def get_image_shapes(path_reg):
    """Returns a dict of image counts for each size. """
    image_files = glob.glob(path_reg)
    sizes = {}
    for image_file in tqdm(image_files):
        img = cv2.imread(image_file, 0)
        if img is None:
            continue
        res = f"{img.shape[0]}x{img.shape[1]}"
        if res in sizes.keys():
            sizes[res] += 1
        else:
            sizes[res] = 1

    # sort the dict by counts
    sizes = {k: v for k, v in sorted(sizes.items(), key=lambda item: item[1], reverse=True)}
    return sizes


def get_total_size(path_reg):
    """Compute the total size of the specified files """
    image_files = glob.glob(path_reg)
    return human_readable_size(sum([os.path.getsize(file) for file in image_files]))

def get_sharpness_and_entropy(path_list, filename="sharpness_entropy.png", zoom=0.1, n=1000):
    """Compute sharpness and entropy of the images in the list.
    make a scatter plot of the two values
    It collects exactly n images, and skips images that are degenerate or unable to be read.
    """
    n = min(n, len(path_list))
    sharpness, entropy = np.zeros(n), np.zeros(n)
    img_list = []
    # get a random permutation of the indices
    indices = torch.randperm(len(path_list))
    pbar = tqdm(total=n)
    for j in indices:
        i = len(img_list)
        if i >= n:
            # we have enough images
            break
        image = cv2.imread(path_list[j])
        image = crop_black_border(image)
        if image is None:
            continue
        pbar.update(1)
        img_list.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sharpness[i] = spatial_variance(gray_image)
        entropy[i] = shannon_entropy(gray_image)

    image_scatter_plot(img_list, sharpness, entropy,
                       zoom=zoom, filename=filename,
                       featurex= "Sharpness", featurey="Entropy")


if __name__ == "__main__":
    fig_dir = os.path.join(configs.fig_dir, "data_exploration")
    os.makedirs(fig_dir, exist_ok=True)

    image_types = {
        "curiosity_mast_color_small": configs.curiosity_mast_color_small,
        "curiosity_mast_color_large": configs.curiosity_mast_color_large,
        "curiosity_navcam_gray": configs.curiosity_navcam_gray,
        "perseverance_mast_color": configs.perseverance_mast_color,
        "perseverance_navcam_color": configs.perseverance_navcam_color
    }
    for name, conf in image_types.items():
        # print(name, get_total_size(conf))
        # print(get_image_shapes(conf))
        zoom = 0.1 if "small" in name else 0.01
        get_sharpness_and_entropy(
            glob.glob(conf),
            os.path.join(fig_dir, f"{name}.png"),
            zoom=zoom)

    # create_image_samples("curiosity")
    # create_image_samples("perseverance")

    # perseverance_mast_color {'1200x1648': 1767, '128x128': 16, '720x1600': 67, '1200x1600': 2, '256x256': 280, '640x272': 2, '1184x1584': 21, '128x320': 1}
    # perseverance_navcam_color {'960x1280': 151, '968x1288': 785, '968x1296': 27, '976x1288': 26, '976x1296': 225}
    # curiosity_mast_color_large {'1200x1200': 1272, '1152x1536': 94, '1200x1648': 27, '512x1024': 8, '1152x1152': 153, '1200x1600': 263, '400x528': 1, '432x688': 1, '1152x1408': 166, '944x1152': 9, '528x720': 8, '512x512': 19, '448x480': 2, '320x384': 6, '640x640': 18, '1200x1408': 218, '352x352': 2, '416x1040': 407, '416x768': 2, '1200x1344': 1431, '432x1152': 533, '400x896': 9, '400x608': 4, '432x560': 3, '640x1344': 2, '432x800': 9, '432x1024': 69, '768x768': 36, '528x512': 1}
    # curiosity_mast_color_small {'144x192': 1194, '144x144': 1904, '48x80': 344, '64x128': 56, '48x64': 8, '144x176': 457, '48x48': 9, '112x144': 53, '32x48': 47, '64x64': 44, '80x80': 27, '32x32': 23, '48x128': 490, '48x96': 11, '16x16': 1, '144x160': 1752, '48x144': 554, '48x112': 9, '80x160': 2, '96x96': 39}
