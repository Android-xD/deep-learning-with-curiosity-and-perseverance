import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from src.utils.utils import batch2img_list
import torch


def plot_images(imgs, titles=None, cmaps='gray', dpi=100, pad=.5,
                adaptive=True, hpad=0.):
    """Plot a set of images horizontally.
    Args:
        imgs: a list of NumPy or PyTorch images, RGB (H, W, 3) or mono (H, W).
        titles: a list of strings, as titles for each image.
        cmaps: colormaps for monochrome images.
        adaptive: whether the figure size should fit the image aspect ratios.
    """
    n = len(imgs)
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * n

    if adaptive:
        ratios = [i.shape[1] / i.shape[0] for i in imgs]  # W / H
    else:
        ratios = [4 / 3] * n
    figsize = [sum(ratios) * 4.5, 4.5 + hpad]
    fig, ax = plt.subplots(
        1, n, figsize=figsize, dpi=dpi, gridspec_kw={'width_ratios': ratios})
    if n == 1:
        ax = [ax]
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmaps[i]))
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        ax[i].set_axis_off()
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
        if titles:
            ax[i].set_title(titles[i], fontsize=25)
    fig.tight_layout(pad=pad)


def plot_dataset(dataset, rows, cols, filename=None):
    """Sample randomly rows*cols images from a dataset and plot them in a grid."""
    imgs = []
    perm = torch.randperm(len(dataset))
    indices = perm[:(rows * cols)]
    for i in indices:
        sample, _ = dataset[i]
        img = batch2img_list(sample.unsqueeze(0), 1)[0]
        imgs.append(img)

    # Create a grid layout for the images
    fig, axes = plt.subplots(rows, cols, figsize=(16, int(16 * rows / cols)))

    for i, ax in enumerate(axes.flat):
        ax.imshow(imgs[i], cmap='gray')
        ax.axis('off')
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=600, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_image_grid(img_lists, titles=None, cmaps='gray', dpi=100, pad=.5, adaptive=False):
    """
    Takes a list of lists of images. Plots each list of images in a row with the corresponding title on the right side.

    Args:
        img_lists: A list of lists, where each inner list contains images.
        titles: A list of strings, with titles for each row of images.
        cmaps: Colormaps for monochrome images.
        dpi: Dots per inch for the figure.
        pad: Padding between rows.
        adaptive: Whether the figure size should fit the image aspect ratios.
    """
    n = len(img_lists)
    m = len(img_lists[0])  # Assuming all inner lists have the same length

    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * (n * m)

    if adaptive:
        # Calculate aspect ratios for each row based on the first list
        ratios = [img_lists[0][i].shape[1] / img_lists[0][i].shape[0] for i in range(m)]
    else:
        # Assuming a default aspect ratio of 4:3
        ratios = [4 / 3] * m

    figsize = [sum(ratios) * 4.5, n * 4.5]
    fig, axs = plt.subplots(n, m, figsize=figsize, dpi=dpi, gridspec_kw={'width_ratios': ratios})

    for i in range(n):
        for j in range(m):
            img = img_lists[i][j]
            axs[i, j].imshow(img, cmap=plt.get_cmap(cmaps[i * m + j]))
            axs[i, j].get_yaxis().set_ticks([])
            axs[i, j].get_xaxis().set_ticks([])
            axs[i, j].set_axis_off()
            for spine in axs[i, j].spines.values():
                spine.set_visible(False)

    if titles:
        for i in range(n):
            axs[i, 0].set_title(titles[i], fontsize=25)

    fig.tight_layout(pad=pad)


def image_scatter_plot(img_list, x, y, zoom=1, filename=None, featurex="Feature 1", featurey="Feature 2"):
    """ Scatter plot with images instead of points
    Args:
        img_list: list of images
        x: x coordinates
        y: y coordinates
        zoom: zoom factor for the images
        filename: if not None, save the figure to this path
        featurex: label for the x-axis
        featurey: label for the y-axis
    """

    def getImage(img):
        # img = cv2.resize(img,(100,100))
        return OffsetImage(img, zoom=zoom)

    fig, ax = plt.subplots()
    ax.set_xlabel(featurex)  # Set x-axis label
    ax.set_ylabel(featurey)  # Set y-axis label

    ax.scatter(x, y, s=0.01)
    for x0, y0, img in zip(x, y, img_list):
        ab = AnnotationBbox(getImage(img), (x0, y0), frameon=False, )
        ax.add_artist(ab)

    if filename:
        plt.savefig(filename, dpi=600, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
