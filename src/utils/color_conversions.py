from skimage.color import rgb2lab, lab2rgb
import torch


def rgb2lab_torch(image):
    """   
    The color image consists of the 'a' and 'b' parts of the LAB format.
    a,b = image[1:, :, :]
    The gray image consists of the `L` part of the LAB format.
    gray_image = image[0, :, :].unsqueeze(0)
    """
    return torch.from_numpy(rgb2lab(image.permute(1, 2, 0))).permute(2, 0, 1)

def lab2rgb_torch(image):
    """
    The color image consists of the 'a' and 'b' parts of the LAB format. and L
    """
    lower = torch.tensor([0, -127, -127]).resize(3, 1, 1)
    upper = torch.tensor([100, 128, 128]).resize(3, 1, 1)
    # clip to valid range
    image = torch.clip(image, lower, upper)
    rgb = torch.from_numpy(lab2rgb(image.permute(1, 2, 0))).permute(2, 0, 1)
    return rgb