import torch
import numpy as np
import cv2


def batch2img_list(tensor, n_max):
    """takes tensor of shape
    B,C,H,W and creates a list of maximum n_max images"""
    tensor = tensor[:n_max].cpu().detach()
    tensor = torch.permute(tensor, (0, 2, 3, 1))
    return [img.numpy() for img in tensor]


def human_readable_size(size, decimal_places=2):
    """https://stackoverflow.com/questions/1094841/get-human-readable-version-of-file-size"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB']:
        if size < 1024.0 or unit == 'PiB':
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f} {unit}"


def to_uint8(array):
    array -= np.min(array)
    array *= 256 / np.max(array)
    return array.astype(np.uint8)


def is_color(img):
    """
    if diff is all zero it means all channels have the same values -> gray
    if any value is not zero, it is a color image
    """
    return np.diff(img, axis=2).any()


def shannon_entropy(gray_image):
    """
    Computes the shannon entropy of an image in grayscale.

    H = -sum(p(x) * log2(p(x)))
    """
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    hist = hist / np.sum(hist)
    return -np.sum(hist * np.log2(hist + 1e-7))


def spatial_variance(gray_image, ksize=9):
    """
    Measures the sharpness of an image.
    Computes the spatial variance of an image in grayscale.
    as E[X^2] - E[X]^2
    """
    exp = cv2.GaussianBlur(gray_image, (ksize, ksize), 0)
    exp = exp.astype(np.float32) ** 2
    x2 = cv2.GaussianBlur(gray_image.astype(np.float32) ** 2, (ksize, ksize), 0)

    var = x2 - exp

    return np.median(var)


def crop_black_border(img, black_threshold=80):
    """
    Cut the black border of an image.
    """
    gray = np.max(img, axis=2)
    row_mean = np.mean(gray, axis=0)
    col_mean = np.mean(gray, axis=1)
    inside_row = np.argwhere(row_mean > black_threshold)
    inside_col = np.argwhere(col_mean > black_threshold)

    if inside_col.shape[0] == 0 or inside_row.shape[0] == 0:
        # the whole image is black
        return img

    x0, x1 = np.min(inside_row), np.max(inside_row)
    y0, y1 = np.min(inside_col), np.max(inside_col)

    return img[y0:y1, x0:x1]


def is_not_degenerate(path, only_color=True, min_entropy=3, min_size=128, min_spatial_variance=0):
    """ Returns True if the image satisfies the constraints.
        """
    image = cv2.imread(path)
    if image is None:
        return False  # corrupted image

    gray_image = cv2.imread(path, 0)
    gray_image = crop_black_border(gray_image)
    if only_color and not is_color(image):
        return False
    if min_size and gray_image.shape[0] < min_size or gray_image.shape[1] < min_size:
        return False
    if shannon_entropy(gray_image) < min_entropy:
        return False
    if spatial_variance(gray_image) < min_spatial_variance:
        return False
    return True
