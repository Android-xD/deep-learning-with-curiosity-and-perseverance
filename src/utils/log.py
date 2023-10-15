import pandas as pd
import seaborn as sns
import json
import os
import glob
import matplotlib.pyplot as plt
"""
The log is implemented as a list of dicts like

{"Epoch": epoch, "Batch": i, "Type": "Train Loss", "Value": loss.item()}
{"Epoch": epoch, "Batch": i, "Type": "Test Loss", "Value": loss.item()}

 We then provide functions that read out relevant information from the log and make plots.
"""

def epoch_report(log, epoch):
    """Filters the list of dictionary's and extracts the mean for a given epoch and type."""
    df = pd.DataFrame(log)
    # filter for epoch
    df = df[(df["Epoch"] == epoch)]
    # compute mean for each type
    df = df.groupby("Type").mean()
    # write as f string
    return f"Epoch {epoch}: " + ", ".join([f"{k}: {v:.4f}" for k, v in df["Value"].items()])


def plot_losses(log, filer_types=None, filename=None):
    df = pd.DataFrame(log)
    if filer_types is not None:
        df = df[df["Type"].isin(filer_types)]

    # Create a line plot with Seaborn
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Epoch", y="Value", hue="Type")
    plt.title("Training and Test Loss Over Epochs")
    # add legend for the confidence interval

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    # set x-axis to integer ticks only
    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
    plt.legend(title="Loss Type")
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=600, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

