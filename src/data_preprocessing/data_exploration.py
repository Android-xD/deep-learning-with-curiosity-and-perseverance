import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import src.utils.configs as configs
from src.utils.utils import human_readable_size
from src.utils.visualization import plot_images


def create_image_samples(rover="curiosity"):
    """Plot a few image samples for each camera type. """
    rover_dir = os.path.join("..", "..", "data", rover)
    camera_types = os.listdir(rover_dir)
    camera_types = [name.replace("LEFT", "*") for name in camera_types]
    camera_types = [name.replace("RIGHT", "*") for name in camera_types]
    camera_types = set(camera_types)

    for i, cam_type in enumerate(camera_types):
        img_list = glob.glob(os.path.join(rover_dir, cam_type, "*"))
        samples = np.arange(len(img_list))
        np.random.shuffle(samples)
        imgs = []
        for i in samples:
            img = cv2.imread(img_list[i])
            if not img is None:
                imgs.append(img[:, :, ::-1])
            if len(imgs) > 10:
                break

        plot_images(imgs)
        path = os.path.join("..", "..", "figures", rover, cam_type + ".png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        path = path.replace("_*", "")
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
    return sizes


def get_total_size(path_reg):
    """Compute the total size of the specified files """
    image_files = glob.glob(path_reg)
    total_size = 0
    for image_file in tqdm(image_files):
        # Read the image using cv2 and compute its size
        image = cv2.imread(image_file)
        if image is not None:
            total_size += image.nbytes  # Get the size in bytes of the image data
    return total_size


if __name__ == "__main__":
    image_types = {
        "curiosity_mast_color_small": configs.curiosity_mast_color_small,
        "curiosity_mast_color_large": configs.curiosity_mast_color_large,
        "perseverance_mast_color": configs.perseverance_mast_color,
        "perseverance_navcam_color": configs.perseverance_navcam_color
    }
    for name, conf in image_types.items():
        print(name, human_readable_size(get_total_size(conf)))
        print(get_image_shapes(conf))

    # create_image_samples("curiosity")
    # create_image_samples("perseverance")

    # perseverance_mast_color {'1200x1648': 1767, '128x128': 16, '720x1600': 67, '1200x1600': 2, '256x256': 280, '640x272': 2, '1184x1584': 21, '128x320': 1}
    # perseverance_navcam_color {'960x1280': 151, '968x1288': 785, '968x1296': 27, '976x1288': 26, '976x1296': 225}
    # curiosity_mast_color_large {'1200x1200': 1272, '1152x1536': 94, '1200x1648': 27, '512x1024': 8, '1152x1152': 153, '1200x1600': 263, '400x528': 1, '432x688': 1, '1152x1408': 166, '944x1152': 9, '528x720': 8, '512x512': 19, '448x480': 2, '320x384': 6, '640x640': 18, '1200x1408': 218, '352x352': 2, '416x1040': 407, '416x768': 2, '1200x1344': 1431, '432x1152': 533, '400x896': 9, '400x608': 4, '432x560': 3, '640x1344': 2, '432x800': 9, '432x1024': 69, '768x768': 36, '528x512': 1}
    # curiosity_mast_color_small {'144x192': 1194, '144x144': 1904, '48x80': 344, '64x128': 56, '48x64': 8, '144x176': 457, '48x48': 9, '112x144': 53, '32x48': 47, '64x64': 44, '80x80': 27, '32x32': 23, '48x128': 490, '48x96': 11, '16x16': 1, '144x160': 1752, '48x144': 554, '48x112': 9, '80x160': 2, '96x96': 39}
