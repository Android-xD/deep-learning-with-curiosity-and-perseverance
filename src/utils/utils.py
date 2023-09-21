import torch

def batch2img_list(tensor, n_max):
    """takes tensor of shape
    B,C,H,W and creates a list of maximum n_max images"""
    tensor = tensor[:n_max].cpu().detach()
    tensor = torch.permute(tensor, (0, 2, 3, 1))
    tensor = torch.clip(tensor,0,1)
    return [img.numpy() for img in tensor]


def human_readable_size(size, decimal_places=2):
    """https://stackoverflow.com/questions/1094841/get-human-readable-version-of-file-size"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB']:
        if size < 1024.0 or unit == 'PiB':
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f} {unit}"
