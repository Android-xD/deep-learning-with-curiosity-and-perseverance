import sys
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm
import os

from src.utils.visualization import plot_images, plot_image_pairs
from src.utils.utils import batch2img_list
from src.utils.configs import trained_weights_dir
from src.data_preprocessing.dataset import ImageDataset
from src.models.autoencoder import *
from src.utils.set_random_seeds import set_seeds
from src.utils import configs
from src.data_preprocessing.dataset import train_test_dataloader, get_transform
from src.utils.log import epoch_report, plot_losses, plot_losses_batch

def get_args():
    parser = argparse.ArgumentParser(description='Trainer for autoencoder')
    parser.add_argument('--experiment', default='indices', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    set_seeds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log = []
    args = get_args()
    experiment = args.experiment

    fig_dir = os.path.join(configs.fig_dir, "ae", experiment)
    os.makedirs(fig_dir, exist_ok=True)

    # Hyperparameters
    batch_size = 8
    learning_rate = 0.005
    num_epochs = 20
    input_size = 16
    dataset_name = "perseverance_navcam_color"
    if experiment == "simple":
        ae = AE(3, 16, [8, 16, 32], [8, 16, 32]).to(device)
    elif experiment == "indices":
        ae = Autoencoder().to(device)
        input_size = 32
    else:
        # exeption
        raise NotImplementedError

    # Data preprocessing and transformation
    transform = get_transform(input_size,True)

    # Load the dataset
    dataset = ImageDataset(dataset_name, transform=transform)
    train_dataloader, test_dataloader = train_test_dataloader(dataset, batch_size)


    train_loss, test_loss = torch.zeros(num_epochs), torch.zeros(num_epochs)


    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(list(ae.parameters()), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        for i, (data, _) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            # move the data to the GPU
            data = data.to(device)
            # Forward pass
            decoded = ae(data.to(device))
            # Compute the loss
            loss = criterion(decoded, data)
            log.append({"Epoch": epoch, "Batch": i, "Type": f"Train MSE Loss", "Value": loss.item()})

            if i == 0:
                input_list = batch2img_list(data, 8)
                output_list = batch2img_list(decoded, 8)
                plot_image_pairs(input_list, output_list, os.path.join(fig_dir, f"train_{epoch:02d}.png"))

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()


        with torch.no_grad():
            for i, (data, _) in enumerate(tqdm(test_dataloader)):
                # move the data to the GPU
                data = data.to(device)
                # Forward pass
                decoded = ae(data.to(device))
                # Compute the loss
                loss = criterion(decoded, data)
                log.append({"Epoch": epoch, "Batch": i, "Type": f"Test MSE Loss", "Value": loss.item()})
                if i == 0:
                    input_list = batch2img_list(data, 8)
                    output_list = batch2img_list(decoded, 8)
                    plot_image_pairs(input_list, output_list, os.path.join(fig_dir, f"test_{epoch:02d}.png"))
        print(epoch_report(log, epoch))
    plot_losses(log, filer_types=["Test MSE Loss", "Train MSE Loss"], filename=os.path.join(fig_dir, f"MSE_losses.png"))
    plot_losses_batch(log, filer_types=["Test MSE Loss", "Train MSE Loss"], filename=os.path.join(fig_dir, f"MSE_losses_batch.png"))

    # Save the trained model
    state_dict = {
        "model_state": ae.state_dict(),
        "log": log,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "dataset_name": dataset_name,
        "input_size": input_size,
    }
    torch.save(state_dict, os.path.join(trained_weights_dir, f'autoencoder_{experiment}.pth'))

