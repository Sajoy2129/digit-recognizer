"""Tests for src/data.py."""

import pytest
import torch

from src.data import VAL_SIZE, count_labels, get_dataloaders, load_datasets


# scope="module" loads the data once and shares it across all tests here.
@pytest.fixture(scope="module")
def datasets():
    return load_datasets()


@pytest.fixture(scope="module")
def train_batch():
    train_loader, _, _ = get_dataloaders(batch_size=64)
    return next(iter(train_loader))


def test_split_sizes(datasets):
    train, val, test = datasets
    assert len(train) == 60_000 - VAL_SIZE
    assert len(val) == VAL_SIZE
    assert len(test) == 10_000


def test_batch_shapes(train_batch):
    images, labels = train_batch
    assert images.shape == (64, 1, 28, 28)
    assert labels.shape == (64,)


def test_labels_are_digits(train_batch):
    _, labels = train_batch
    assert labels.min() >= 0
    assert labels.max() <= 9


def test_images_are_normalized(train_batch):
    images, _ = train_batch
    # After normalization the pixel mean should be close to 0 (not 0.13).
    assert abs(images.mean().item()) < 0.2
    assert images.dtype == torch.float32


def test_split_is_reproducible():
    _, val_a, _ = load_datasets()
    _, val_b, _ = load_datasets()
    assert val_a.indices == val_b.indices


def test_count_labels_sums_to_dataset_size(datasets):
    train, _, _ = datasets
    counts = count_labels(train)
    assert len(counts) == 10
    assert sum(counts) == len(train)
