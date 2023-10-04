import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


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
        ratios = [4/3] * n
    figsize = [sum(ratios)*4.5, 4.5+ hpad]
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
            ax[i].set_title(titles[i],fontsize=25)
    fig.tight_layout(pad=pad)


def image_scatter_plot(img_list, x, y, zoom=1):
    """ Scatter plot with images instead of points
    Args:
        img_list: list of images
        x: x coordinates
        y: y coordinates
    """
    def getImage(img):
        # img = cv2.resize(img,(100,100))
        return OffsetImage(img, zoom=zoom)

    fig, ax = plt.subplots()
    ax.set_xlabel("Feature 1")  # Set x-axis label
    ax.set_ylabel("Feature 2")  # Set y-axis label

    ax.scatter(x, y)
    for x0, y0, img in zip(x, y, img_list):
        ab = AnnotationBbox(getImage(img), (x0, y0), frameon=False)
        ax.add_artist(ab)

    # fig.savefig("overlap.png", dpi=600, bbox_inches="tight")
    plt.show()
