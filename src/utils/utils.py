import torch

def batch2img_list(tensor, n_max):
    """takes tensor of shape
    B,C,H,W and creates a list of maximum n_max images"""
    tensor = tensor[:n_max].cpu().detach()
    tensor = torch.permute(tensor, (0, 2, 3, 1))
    tensor = torch.clip(tensor,0,1)
    return [img.numpy() for img in tensor]