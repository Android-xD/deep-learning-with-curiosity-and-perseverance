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
    However, we need to have a threshold because of transmission errors
    that introduce color smears (probably).
    """
    return np.mean(np.diff(img.astype(float), axis=2)) > 3


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
    Measures the sharpness of an image as local color variance.
    The resulting score has the same range as the pixel values squared. [0, 255^2]
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
        return None

    x0, x1 = np.min(inside_row), np.max(inside_row)
    y0, y1 = np.min(inside_col), np.max(inside_col)

    # check if the image is too small
    if x0 >= x1-16 or y0 >= y1-16:
        return None

    return img[y0:y1, x0:x1]


def is_not_degenerate(path, only_color=True, min_entropy=2, min_size=64, min_spatial_variance=5):
    """ Returns the cropped image if the image satisfies the constraints.
        Otherwise, returns None
        Minimum size is set to 64 to mitigate the compression artifacts of the small images.
        """
    image = cv2.imread(path)
    if image is None:
        return None  # corrupted image

    image = crop_black_border(image)

    if image is None:
        return None  # black image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if only_color and not is_color(image):
        return None
    if min_size and gray_image.shape[0] < min_size or gray_image.shape[1] < min_size:
        return None
    if shannon_entropy(gray_image) < min_entropy:
        return None
    if spatial_variance(gray_image) < min_spatial_variance:
        return None
    return image
