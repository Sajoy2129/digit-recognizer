"""Tests for src/train.py and src/evaluate.py.

These use small fake datasets, so they run fast and need no download.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluate import evaluate
from src.models import MLP
from src.train import get_device, train_one_epoch

CPU = torch.device("cpu")


def make_fake_loader(size: int = 32) -> DataLoader:
    torch.manual_seed(0)
    images = torch.randn(size, 1, 28, 28)
    labels = torch.randint(0, 10, (size,))
    return DataLoader(TensorDataset(images, labels), batch_size=size)


def test_training_lowers_the_loss():
    """Sanity check: a few steps on one tiny batch should reduce the loss."""
    loader = make_fake_loader()
    model = MLP()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    first_loss = train_one_epoch(model, loader, loss_fn, optimizer, CPU)
    for _ in range(10):
        last_loss = train_one_epoch(model, loader, loss_fn, optimizer, CPU)
    assert last_loss < first_loss


def test_evaluate_returns_fraction_between_0_and_1():
    accuracy = evaluate(MLP(), make_fake_loader(), CPU)
    assert 0.0 <= accuracy <= 1.0


def test_evaluate_is_1_for_a_perfect_model():
    class AlwaysCorrect(nn.Module):
        """Cheats: 'predicts' the label hidden in the first pixel."""

        def forward(self, x):
            labels = x[:, 0, 0, 0].long()
            return nn.functional.one_hot(labels, num_classes=10).float()

    images = torch.zeros(20, 1, 28, 28)
    labels = torch.arange(20) % 10
    images[:, 0, 0, 0] = labels.float()
    loader = DataLoader(TensorDataset(images, labels), batch_size=5)
    assert evaluate(AlwaysCorrect(), loader, CPU) == 1.0


def test_get_device_returns_a_device():
    assert isinstance(get_device(), torch.device)
