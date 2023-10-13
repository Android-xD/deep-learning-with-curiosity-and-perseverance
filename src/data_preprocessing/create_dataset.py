"""
This module creates a filtered dataset from a list of image files.

The image files are filtered according to the is_degenerate function.

for small images:
only_color=True, min_entropy=4.75, min_size=128, min_spatial_variance=15
for large images:
only_color=True, min_entropy=4.75, min_size=512, min_spatial_variance=15

the images are cropped to remove the black border and copied to the dataset directory.
"""
import glob
import os
import cv2
from tqdm import tqdm
import src.utils.configs as configs
from src.utils.utils import human_readable_size
from src.utils.utils import is_not_degenerate
from threading import Thread


def copy_image(image_files, dataset_name, filter_params={}, pbar=None):
    """ Copies a list of image files to the dataset directory if they satisfiy the constraints. """
    for file in image_files:
        res = is_not_degenerate(file, **filter_params)
        if not res is None:
            out_file = os.path.join(configs.data_dir, "datasets", dataset_name, os.path.basename(file))
            cv2.imwrite(out_file, res)
        if not pbar is None:
            pbar.update(1)
def create_dataset(image_files, dataset_name, filter_params={}, num_threads=8):
    """
    Create a dataset from a list of image files.
    The image files are filtered according to the filter_params.
    """
    os.makedirs(os.path.join(configs.data_dir, "datasets", dataset_name), exist_ok=True)

    print(f"Found {len(image_files)} images for {dataset_name}")
    print(f"Total size: {human_readable_size(sum([os.path.getsize(file) for file in image_files]))}")

    threads = []
    pbar = tqdm(total=len(image_files), desc=dataset_name)
    for i in range(num_threads):
        thread = Thread(target= lambda : copy_image(image_files[i::num_threads], dataset_name, filter_params, pbar))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    filtered_image_files = glob.glob(os.path.join(configs.data_dir, "datasets", dataset_name, "*"))
    print(f"{len(filtered_image_files)} remaining from {len(image_files)}")
    print(f"Total size: {human_readable_size(sum([os.path.getsize(file) for file in filtered_image_files]))}")


if __name__ == "__main__":
    small_filter_params = {"only_color": True, "min_entropy": 4.75, "min_size": 128, "min_spatial_variance": 15}
    large_filter_params = {"only_color": True, "min_entropy": 4.75, "min_size": 512, "min_spatial_variance": 15}
    small_filter_gray_params = {"only_color": False, "min_entropy": 4.75, "min_size": 128, "min_spatial_variance": 15}

    datasets = {
        "curiosity_mast_color_small": (configs.curiosity_mast_color_small, small_filter_params),
        "curiosity_mast_color_large": (configs.curiosity_mast_color_large, large_filter_params),
        "curiosity_navcam_gray": (configs.curiosity_navcam_gray, small_filter_gray_params),
        "perseverance_mast_color": (configs.perseverance_mast_color, large_filter_params),
        "perseverance_navcam_color": (configs.perseverance_navcam_color, large_filter_params),
    }
    for dataset_name, (image_file_pattern, params) in datasets.items():
        image_files = glob.glob(image_file_pattern)
        create_dataset(image_files, dataset_name, params)