import pandas as pd
import seaborn as sns
import json
import os
import glob
import matplotlib.pyplot as plt
import re


def read_log(log_file, fine=False):
    """
    The log is implemented as a list of dicts like

    {"Epoch": epoch, "Batch": i, "Type": "Train Loss", "Value": loss.item()}
    {"Epoch": epoch, "Batch": i, "Type": "Test Loss", "Value": loss.item()}

     We read out relevant information from the log and make plots.
    """
    loss_list = []

    with open(log_file, 'r') as file:
        lines = file.readlines()
        epoch = 0
        for line in lines:
            epoch_match = re.search(r'Epoch: \[(\d+)\]\[(\d+)\/(\d+)\].+Loss (\d+\.\d+)', line)
            if epoch_match and not fine:
                epoch = int(epoch_match.group(1))
                batch = int(epoch_match.group(2))
                total_batches = int(epoch_match.group(3))
                loss = float(epoch_match.group(4))

                train_loss_entry = {"Epoch": epoch, "Batch": batch, "Type": "Train Loss", "Value": loss}
                #test_loss_entry = {"Epoch": epoch, "Batch": batch, "Type": "Test Loss", "Value": loss}

                loss_list.append(train_loss_entry)
                #loss_list.append(test_loss_entry)
            else:
                # Handle the alternative format
                alt_epoch_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+).+Loss (\d+\.\d+)', line)
                if alt_epoch_match and fine:
                    epoch +=1
                    loss = float(alt_epoch_match.group(2))

                    entry = {"Epoch": epoch, "Type": "Train Loss", "Value": loss}
                    loss_list.append(entry)

    return loss_list

def epoch_report(log, epoch):
    """Filters the list of dictionary's and extracts the mean for a given epoch and type."""
    df = pd.DataFrame(log)
    # filter for epoch
    df = df[(df["Epoch"] == epoch)]
    # compute mean for each type
    df = df.groupby("Type").mean()
    # write as f string
    return f"Epoch {epoch}: " + ", ".join([f"{k}: {v:.4f}" for k, v in df["Value"].items()])


def plot_losses(log, filer_types=None, filename=None, y_label=""):
    df = pd.DataFrame(log)
    if filer_types is not None:
        df = df[df["Type"].isin(filer_types)]

    # Create a line plot with Seaborn
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    # confidence interval 95%
    sns.lineplot(data=df, x="Epoch", y="Value", hue="Type", errorbar=("se", 2))
    plt.title("Training and Test Loss Over Epochs")
    # add legend for the confidence interval

    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    # set x-axis to integer ticks only
    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
    plt.legend(title="Loss Type")
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=600, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_losses_batch(log, filer_types=None, filename=None, y_label=""):
    df = pd.DataFrame(log)
    if filer_types is not None:
        df = df[df["Type"].isin(filer_types)]

    # Create a line plot with Seaborn
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    # compute epoch from batch
    for t in df["Type"].unique():
        batches_per_epoch = len(df[df["Type"] == t]["Batch"].unique())
        df.loc[df["Type"] == t, "Epoch"] += df[df["Type"] == t]["Batch"] / batches_per_epoch


    sns.lineplot(data=df, x="Epoch", y="Value", hue="Type")
    plt.title("Training and Test Loss Over Epochs")
    # add legend for the confidence interval

    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    # set x-axis to integer ticks only
    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
    plt.legend(title="Loss Type")
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=600, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

if __name__ == '__main__':
    log = read_log("../../logs/imagenet-autoencoder_vgg16.err", fine=False)
    plot_losses(log, filename="../../figures/imagenet-autoencoder/loss_imagenet-autoencoder_vgg16.png")
    log = read_log("../../logs/fine-imagenet-autoencoder.err", fine=True)
    plot_losses(log, filename="../../figures/imagenet-autoencoder/loss_fine-imagenet-autoencoder.png")