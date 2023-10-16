import matplotlib.pyplot as plt
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from src.data_preprocessing.dataset import ImageDataset
from src.utils import configs
from torchvision import transforms
from src.utils.set_random_seeds import set_seeds
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from src.utils.visualization import plot_images, plot_dataset
from src.utils.utils import batch2img_list
from src.data_preprocessing.dataset import train_test_dataloader
import os
from src.utils.configs import trained_weights_dir
from src.utils.log import epoch_report, plot_losses

if __name__ == '__main__':
    set_seeds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log = []
    fig_dir = os.path.join(configs.fig_dir, "diffusion")
    os.makedirs(fig_dir, exist_ok=True)

    # Hyperparameters
    batch_size = 8
    learning_rate = 0.0005
    num_epochs = 10
    input_size = 64
    dataset_name = "curiosity_mast_color_small"  # "perseverance_mast_color"

    # Data preprocessing and transformation
    transform = transforms.Compose([
        transforms.Resize(2 * input_size),
        transforms.RandomCrop((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        flash_attn=True
    )

    diffusion = GaussianDiffusion(
        model,
        image_size=input_size,
        timesteps=1000  # number of steps
    )

    dataset = ImageDataset(dataset_name, transform=transform)
    plot_dataset(dataset, 5, 7)
    plt.show()
    train_dataloader, _ = train_test_dataloader(dataset, batch_size=batch_size)

    optimizer = torch.optim.Adam(diffusion.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for i, (data, _) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()

            loss = diffusion(data)
            loss.backward()

            log.append({"Epoch": epoch, "Batch": i, "Type": "Train Loss", "Value": loss.item()})

            optimizer.step()
        print(epoch_report(log, epoch))

    # plot_losses(log)
    plt.savefig(os.path.join(fig_dir, f"diffusion_loss.png"), dpi=600, bbox_inches="tight")

    # Save the trained model
    state_dict = {
        "model_state": diffusion.state_dict(),
        "log": log,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "dataset_name": dataset_name,
        "input_size": input_size,
    }
    torch.save(state_dict, os.path.join(trained_weights_dir, 'diffusion.pth'))

    with torch.inference_mode():
        for i in range(20):
            sampled_images = diffusion.sample(batch_size=batch_size)
            plot_images(batch2img_list(sampled_images, batch_size))
            plt.savefig(os.path.join(fig_dir, f"diffusion_{i}.png"), dpi=600, bbox_inches="tight")
