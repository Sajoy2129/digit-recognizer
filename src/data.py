"""Load MNIST and build train/validation/test DataLoaders."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

# Project root is one level above src/, so data lands in digit-recognizer/data/
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Mean and standard deviation of all MNIST training pixels (after scaling to 0-1).
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

VAL_SIZE = 5_000
SEED = 42


def get_transform() -> transforms.Compose:
    """Turn a PIL image into a normalized tensor of shape (1, 28, 28)."""
    return transforms.Compose([
        transforms.ToTensor(),  # pixels 0-255 -> floats 0.0-1.0
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),  # centre around 0
    ])


def load_datasets(
    data_dir: Path = DATA_DIR,
    val_size: int = VAL_SIZE,
    seed: int = SEED,
) -> tuple[Dataset, Dataset, Dataset]:
    """Download MNIST (if needed) and return (train, validation, test) datasets.

    The official 60k training set is split into train and validation.
    The 10k test set is kept separate and only used for the final evaluation.
    """
    transform = get_transform()
    full_train = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    test = datasets.MNIST(
        data_dir, train=False, download=True, transform=transform
    )

    # A seeded generator makes the split identical on every run.
    generator = torch.Generator().manual_seed(seed)
    train_size = len(full_train) - val_size
    train, val = random_split(
        full_train, [train_size, val_size], generator=generator
    )
    return train, val, test


def get_dataloaders(
    batch_size: int = 64,
    data_dir: Path = DATA_DIR,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train, validation, test) DataLoaders that yield batches."""
    train, val, test = load_datasets(data_dir)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size)
    test_loader = DataLoader(test, batch_size=batch_size)
    return train_loader, val_loader, test_loader


def get_labels(dataset: Dataset) -> torch.Tensor:
    """Return all labels of an MNIST dataset (or a Subset of one) as a tensor.

    Reading labels directly is much faster than looping over every image.
    """
    if isinstance(dataset, Subset):
        return get_labels(dataset.dataset)[dataset.indices]
    return dataset.targets


def get_raw_image(dataset: Dataset, index: int) -> torch.Tensor:
    """Return the original 28x28 image as integers 0-255 (no normalization).

    0 is a black background pixel and 255 is a fully white ink pixel.
    """
    if isinstance(dataset, Subset):
        return get_raw_image(dataset.dataset, dataset.indices[index])
    return dataset.data[index]


def get_raw_images(dataset: Dataset) -> torch.Tensor:
    """Return all images of an MNIST dataset (or a Subset) as uint8, (N, 28, 28).

    Like get_labels, this skips the per-image transform, so it is fast.
    """
    if isinstance(dataset, Subset):
        return get_raw_images(dataset.dataset)[dataset.indices]
    return dataset.data


def count_labels(dataset: Dataset) -> list[int]:
    """Return how many examples there are of each digit 0-9."""
    return torch.bincount(get_labels(dataset), minlength=10).tolist()
