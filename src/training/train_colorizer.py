import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

from tqdm import tqdm
import os
import json

import src.utils.configs
from src.utils.visualization import plot_images
from src.utils.utils import batch2img_list
from src.utils.set_random_seeds import set_seeds
from src.utils.configs import trained_weights_dir
from src.data_preprocessing.dataset import CIELABDataset, ImageDataset
from src.utils import configs
from src.models.colorizer import Network6, Network_Prob, get_mle
from src.utils.color_conversions import lab2rgb_torch
from src.utils.log import epoch_report, plot_losses, plot_losses_batch
from src.data_preprocessing.dataset import train_test_dataloader, get_transform


def mse_from_lab(gray, color, pred):
    """
    Compute the mean squared error between two LAB images
    """
    # slice along color channel
    img_orig = lab2rgb_torch(torch.cat((gray, color.detach()), dim=1).cpu()[0]).unsqueeze(0)
    img_pred = lab2rgb_torch(torch.cat((gray, pred.detach()), dim=1).cpu()[0]).unsqueeze(0)
    mse = nn.MSELoss()
    return mse(img_orig, img_pred)


def plot_LAB_prediction(gray, color, pred):
    gray_orig = batch2img_list(gray.cpu(), 1)
    # slice along color channel
    img_orig = batch2img_list(lab2rgb_torch(torch.cat((gray, color.detach()), dim=1).cpu()[0]).unsqueeze(0),
                              1)
    ab_orig = batch2img_list(color[0].unsqueeze(1)+127, 2)
    ab_pred = batch2img_list(pred[0].unsqueeze(1)+127, 2)
    img_pred = batch2img_list(lab2rgb_torch(torch.cat((gray, pred.detach()), dim=1).cpu()[0]).unsqueeze(0),
                              1)
    # apply cmap on gray_orig + ab_orig + ab_pred

    images = img_orig +gray_orig + ab_orig + ab_pred + img_pred
    labels = ["original", "L", "A", "B", "A predicted", "B predicted", "predicted"]
    plot_images(images, labels, hpad=1.5)

def criterion(y_true, y_pred):
    """for classification"""
    cle = nn.CrossEntropyLoss()
    batch_size, num_classes, height, width = y_pred.shape
    # reshape y_pred (batch_size, 2*num_classes, height, width)
    # to (batch_size, num_classes, 2*height, width)
    y_pred = y_pred.reshape(2*batch_size, num_classes//2, height, width)
    # reshape y_true (batch_size, 2, height, width)
    # to (batch_size, 1, 2*height, width)
    y_true = y_true.reshape(2*batch_size,  height, width)+127
    return cle(y_pred, y_true.to(torch.long))



if __name__ == '__main__':
    set_seeds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log = []
    experiment = "Prob"
    fig_dir = os.path.join(configs.fig_dir, f"colorizer_{experiment}")
    os.makedirs(fig_dir, exist_ok=True)

    # Hyperparameters
    batch_size = 8
    learning_rate = 0.00005
    num_epochs = 20
    input_size = 64
    dataset_name = "perseverance_navcam_color"

    # Initialize model
    colorizer = Network_Prob().to(device)

    # Data preprocessing and transformation
    transform = get_transform(input_size, True)

    # Load the dataset
    dataset = CIELABDataset(dataset_name, transform=transform)
    #src.utils.visualization.plot_dataset(dataset, 5, 7)
    #plt.show()

    train_dataloader, test_dataloader = train_test_dataloader(dataset, batch_size)
    mse = nn.MSELoss()
    if experiment == "Reg":
        colorizer = Network6().to(device)
        criterion = nn.MSELoss() # for regression
        get_mle = lambda x:x

    optimizer = torch.optim.Adam(colorizer.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        for i, (gray, color) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            # move the data to the GPU
            gray, color = gray.to(device), color.to(device)
            # Forward pass
            pred = colorizer(gray)

            # Compute the loss
            loss = criterion(color, pred)
            pred = get_mle(pred)

            log.append({"Epoch": epoch, "Batch": i, "Type": "Train Loss", "Value": loss.item()})
            log.append({"Epoch": epoch, "Batch": i, "Type": "Train MSE RGB", "Value": mse_from_lab(gray, color, pred).item()})
            log.append({"Epoch": epoch, "Batch": i, "Type": "Train MSE LAB", "Value": mse(color, pred).item()})
            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            if i < 10:
                plot_LAB_prediction(gray, color, pred)
                plt.savefig(os.path.join(fig_dir, f"train_{epoch:02d}_{i:04d}_output.png"))
                plt.close()

        with torch.no_grad():
            for i, (gray, color) in enumerate(tqdm(test_dataloader)):
                # move the data to the GPU
                gray, color = gray.to(device), color.to(device)
                # Forward pass
                pred = colorizer(gray)

                # Compute the loss
                loss = criterion(color, pred)
                pred = get_mle(pred)
                log.append({"Epoch": epoch, "Batch": i, "Type": "Test Loss", "Value": loss.item()})
                log.append({"Epoch": epoch, "Batch": i, "Type": "Test MSE RGB",
                            "Value": mse_from_lab(gray, color, pred).item()})
                log.append({"Epoch": epoch, "Batch": i, "Type": "Test MSE LAB", "Value": mse(color, pred).item()})

                if i < 10:
                    plot_LAB_prediction(gray, color, pred)
                    plt.savefig(os.path.join(fig_dir, f"test_{epoch:02d}_{i:04d}_output.png"))
                    plt.close()

        print(epoch_report(log, epoch))

    plot_losses(log, filer_types=["Test Loss", "Train Loss"], filename=os.path.join(fig_dir, f"losses.png"))
    plot_losses_batch(log, filer_types=["Test Loss", "Train Loss"], filename=os.path.join(fig_dir, f"losses_batch.png"))
    plot_losses(log, filer_types=["Test MSE RGB", "Train MSE RGB"], filename=os.path.join(fig_dir, f"mse_rgb.png"))
    plot_losses(log, filer_types=["Test MSE LAB", "Train MSE LAB"], filename=os.path.join(fig_dir, f"mse_lab.png"))
    state_dict = {
        "model_state": colorizer.state_dict(),
        "log": log,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "dataset_name": dataset_name,
        "input_size": input_size,
    }

    # Save the trained models
    torch.save(colorizer.state_dict(), os.path.join(trained_weights_dir, f'colorizer_{experiment}.pth'))
