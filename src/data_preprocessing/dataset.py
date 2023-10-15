import glob

from PIL import Image, ImageFile

from torch.utils.data import Dataset, DataLoader, random_split
from src.utils.color_conversions import rgb2lab_torch
from src.utils.configs import data_dir
# fix for truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_test_dataloader(dataset, batch_size, train_fraction=0.9):
    """Makes a train and test loader from a dataset."""
    n_train = int(train_fraction * len(dataset))
    train_dataset, test_dataset = random_split(dataset, [n_train, len(dataset) - n_train])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader


class ImageDataset(Dataset):
    """
        ImageDataset class for loading images from a file pattern
        transform: torchvision.transforms
    """

    def __init__(self, dataset_name, transform=None):
        self.image_files = glob.glob(os.path.join(data_dir, "datasets", dataset_name, "*"))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, 0


class CIELABDataset(Dataset):
    """Idea from https://github.com/gkamtzir/cnn-image-colorization/tree/main"""

    def __init__(self, dataset_name, transform=None):
        self.image_files = glob.glob(os.path.join(data_dir, "datasets", dataset_name, "*"))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # convert to lab
        lab = rgb2lab_torch(image)

        # convert to grayscale
        gray = lab[:1, :, :]
        color = lab[1:, :, :]

        return gray, color


if __name__ == '__main__':
    import os
    from torchvision import transforms

    from src.utils.visualization import plot_dataset
    from src.utils.set_random_seeds import set_seeds
    from src.utils import configs

    fig_dir = os.path.join(configs.fig_dir, "datasets")
    os.makedirs(fig_dir, exist_ok=True)

    dataset_list = os.listdir(os.path.join(data_dir, "datasets"))
    input_size = 128
    transform = transforms.Compose([
        transforms.Resize(2 * input_size),
        transforms.RandomCrop((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    for dataset_name in dataset_list:
        set_seeds()
        dataset = ImageDataset(dataset_name, transform=transform)
        filename = os.path.join(fig_dir, f"{dataset_name}.png")
        plot_dataset(dataset, 5, 7, filename=filename)
