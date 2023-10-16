import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms

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
from src.models.colorizer import Network6
from src.utils.color_conversions import lab2rgb_torch
from src.utils.log import epoch_report, plot_losses
from src.data_preprocessing.dataset import train_test_dataloader

if __name__ == '__main__':
    set_seeds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log = []
    fig_dir = os.path.join(configs.fig_dir, "colorizer")
    os.makedirs(fig_dir, exist_ok=True)

    # Hyperparameters
    batch_size = 8
    learning_rate = 0.0005
    num_epochs = 8
    input_size = 64
    dataset_name = "perseverance_navcam_color"

    # Initialize model
    colorizer = Network6().to(device)

    # Data preprocessing and transformation
    transform = transforms.Compose([
        transforms.Resize(2*input_size),
        transforms.RandomCrop((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    # Load the dataset
    dataset = CIELABDataset(dataset_name, transform=transform)
    #src.utils.visualization.plot_dataset(dataset, 5, 7)
    #plt.show()

    train_dataloader, test_dataloader = train_test_dataloader(dataset, batch_size)

    def criterion(y_true, y_pred):
        mse = nn.MSELoss()
        return mse(y_true, y_pred)


    optimizer = torch.optim.Adam(colorizer.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        for i, (gray, color) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            # move the data to the GPU
            gray, color = gray.to(device), color.to(device)
            # Forward pass
            pred = colorizer(gray)
            pred_img = torch.cat((gray, pred.detach()), dim=1).cpu()

            # Compute the loss
            loss = criterion(color, pred)
            log.append({"Epoch": epoch, "Batch": i, "Type": "Train Loss", "Value": loss.item()})

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                gray_orig = batch2img_list(gray.cpu(), 1)
                # slice along color channel
                img_orig = batch2img_list(lab2rgb_torch(torch.cat((gray, color.detach()), dim=1).cpu()[0]).unsqueeze(0),
                                          1)
                ab_orig = batch2img_list(color[0].unsqueeze(1), 2)
                ab_pred = batch2img_list(pred[0].unsqueeze(1), 2)
                img_pred = batch2img_list(lab2rgb_torch(torch.cat((gray, pred.detach()), dim=1).cpu()[0]).unsqueeze(0),
                                          1)

                images = img_orig + gray_orig + ab_orig + ab_pred + img_pred
                labels = ["original", "L", "A", "B", "A predicted", "B predicted", "predicted"]
                plot_images(images, labels, hpad=1.5)
                plt.savefig(os.path.join(fig_dir, f"train_{epoch:02d}_{i:04d}_output.png"))
                plt.close()

        with torch.no_grad():
            for i, (gray, color) in enumerate(tqdm(test_dataloader)):
                # move the data to the GPU
                gray, color = gray.to(device), color.to(device)
                # Forward pass
                pred = colorizer(gray)
                pred_img = torch.cat((gray, pred.detach()), dim=1)
                # Compute the loss
                loss = criterion(color, pred)
                log.append({"Epoch": epoch, "Batch": i, "Type": "Test Loss", "Value": loss.item()})

        print(epoch_report(log, epoch))

    plot_losses(log)
    plt.savefig(os.path.join(fig_dir, f"MSE_reconstruction_error_{epoch:04d}.png"))
    # save log as json
    json.dump(log, open(os.path.join(fig_dir, f"log_{epoch:04d}.json"), "w"))

    # Save the trained models
    torch.save(colorizer.state_dict(), os.path.join(trained_weights_dir, 'colorizer.pth'))
