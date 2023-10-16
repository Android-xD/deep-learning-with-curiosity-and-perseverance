import torch
import os

import matplotlib.pyplot as plt
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

from src.utils import configs
from src.utils.set_random_seeds import set_seeds
from src.utils.visualization import plot_images
from src.utils.utils import batch2img_list
from src.utils.configs import trained_weights_dir


if __name__ == '__main__':
    set_seeds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fig_dir = os.path.join(configs.fig_dir, "diffusion")
    os.makedirs(fig_dir, exist_ok=True)

    state = torch.load(os.path.join(trained_weights_dir, 'diffusion.pth'))
    input_size = state['input_size']
    batch_size = state['batch_size']


    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        flash_attn=True
    )
    model.load_state_dict(state['model_state'])

    diffusion = GaussianDiffusion(
        model,
        image_size=input_size,
        timesteps=1000  # number of steps
    )

    with torch.inference_mode():
        for i in range(20):
            sampled_images = diffusion.sample(batch_size=batch_size)
            plot_images(batch2img_list(sampled_images, batch_size))
            plt.savefig(os.path.join(fig_dir, f"diffusion_{i}.png"), dpi=600, bbox_inches="tight")
