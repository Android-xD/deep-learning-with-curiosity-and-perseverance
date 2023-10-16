import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm
import os

from src.utils.visualization import plot_images, plot_image_pairs
from src.utils.log import epoch_report, plot_losses
from src.utils.utils import batch2img_list
from src.utils.set_random_seeds import set_seeds
from src.utils.configs import trained_weights_dir
from src.data_preprocessing.dataset import ImageDataset, train_test_dataloader
from src.models.vae import VAE
from src.utils import configs

if __name__ == '__main__':
    set_seeds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log = []
    fig_dir = os.path.join(configs.fig_dir, "vae")
    os.makedirs(fig_dir, exist_ok=True)

    # Hyperparameters
    batch_size = 8
    learning_rate = 0.0005
    num_epochs = 20
    input_size = 64
    dataset_name = "perseverance_navcam_color"

    # Initialize the encoder and decoder
    vae_16 = VAE(in_channels=3, out_channels=32,
                 latent_channels=32,
                 encoder_channels=[32, 64, 128],
                 decoder_channels=[128, 64, 32])

    vae_32 = VAE(in_channels=3, out_channels=64,
                 latent_channels=64,
                 encoder_channels=[32, 64, 128, 256],
                 decoder_channels=[256, 128, 64, 32])

    vae_64 = VAE(in_channels=3, out_channels=128,
                 latent_channels=128,
                 encoder_channels=[32, 64, 128, 256, 512],
                 decoder_channels=[512, 256, 128, 64, 32])

    vae_128 = VAE(in_channels=3, out_channels=512,
                  latent_channels=512,
                  encoder_channels=[32, 64, 128, 256, 512, 1024],
                  decoder_channels=[1024, 512, 256, 128, 64, 32])

    vae = vae_64.to(device)

    # Data preprocessing and transformation
    transform = transforms.Compose([
        transforms.Resize(2 * input_size),
        transforms.RandomCrop((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    # Load the dataset
    dataset = ImageDataset(dataset_name, transform=transform)
    train_dataloader, test_dataloader = train_test_dataloader(dataset, batch_size)

    # Plot the beta schedule
    beta_schedule = torch.linspace(0, 1, num_epochs)
    # Plot the beta schedule
    plt.figure()
    plt.xlabel("epoch")
    plt.ylabel("beta")
    plt.plot(beta_schedule)
    plt.legend(["KL"])
    plt.savefig(os.path.join(fig_dir, f"beta_schedule.png"))


    def criterion(decoded, data, mean, log_var, beta=0.0, split="Train"):
        reconstruction_loss = nn.MSELoss()(decoded, data)
        # Compute the KL divergence term
        # using the formula for the KL divergence of two Gaussian distributions:
        # https://arxiv.org/pdf/1312.6114.pdf
        kl_divergence = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        # scale the divergence to same magnitude as mse
        kl_divergence /= data.shape[0] * data.shape[1] * data.shape[2] * data.shape[3]

        log.append({"Epoch": epoch, "Batch": i, "Type": f"{split} MSE Loss", "Value": reconstruction_loss.item()})
        log.append({"Epoch": epoch, "Batch": i, "Type": f"{split} KL Loss", "Value": kl_divergence.item()})

        # Combine both terms in the loss
        return reconstruction_loss + beta * kl_divergence


    optimizer = torch.optim.Adam(list(vae.parameters()), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        for i, (data, _) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            # move the data to the GPU
            data = data.to(device)
            # Forward pass
            decoded, mean, log_var = vae(data.to(device))
            # Compute the loss
            loss = criterion(decoded, data, mean, log_var, beta_schedule[epoch], "Train")
            log.append({"Epoch": epoch, "Batch": i, "Type": "Train Loss", "Value": loss.item()})

            if i % 1000 == 0:
                input_list = batch2img_list(data, 5)
                output_list = batch2img_list(decoded, 5)
                plot_image_pairs(input_list, output_list)
                plt.savefig(os.path.join(fig_dir, f"train_{epoch:02d}.png"))

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            for i, (data, _) in enumerate(tqdm(test_dataloader)):
                # move the data to the GPU
                data = data.to(device)
                # Forward pass
                reconstruction, mean, log_var = vae(data.to(device))
                # Compute the loss
                loss = criterion(reconstruction, data, mean, log_var, split="Test")
                log.append({"Epoch": epoch, "Batch": i, "Type": "Test Loss", "Value": loss.item()})

                if i == 0:
                    input_list = batch2img_list(data, 5)
                    output_list = batch2img_list(reconstruction, 5)
                    plot_image_pairs(input_list, output_list)
                    plt.savefig(os.path.join(fig_dir, f"test_{epoch:02d}_output.png"))
                    plt.close()

        print(epoch_report(log, epoch))
    plot_losses(log, filer_types=["Test Loss", "Train Loss"], filename=os.path.join(fig_dir, f"beta_losses.png"))
    plot_losses(log, filer_types=["Test MSE Loss", "Train MSE Loss"], filename=os.path.join(fig_dir, f"MSE_losses.png"))
    plot_losses(log, filer_types=["Test KL Loss", "Train KL Loss"], filename=os.path.join(fig_dir, f"KL_losses.png"))
    plot_losses(log, filer_types=None, filename=os.path.join(fig_dir, f"losses.png"))

    # Save the trained model
    state_dict = {
        "model_state": vae.state_dict(),
        "log": log,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "dataset_name": dataset_name,
        "input_size": input_size,
    }
    torch.save(state_dict, os.path.join(trained_weights_dir, 'vae.pth'))
