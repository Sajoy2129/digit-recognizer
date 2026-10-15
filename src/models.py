"""PyTorch model definitions: MLP and CNN."""

import torch
from torch import nn


class MLP(nn.Module):
    """A multi-layer perceptron with one or more hidden layers.

    hidden_sizes=(128,) gives one hidden layer of 128 neurons;
    hidden_sizes=(256, 128) gives two hidden layers.

    Input: a batch of images, shape (batch, 1, 28, 28).
    Output: 10 raw scores ("logits") per image, one per digit.
    There is no softmax here because CrossEntropyLoss applies it internally.
    """

    def __init__(self, hidden_sizes: tuple[int, ...] = (128,)) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten()]  # (batch, 1, 28, 28) -> (batch, 784)
        in_size = 28 * 28
        for size in hidden_sizes:
            layers.append(nn.Linear(in_size, size))
            layers.append(nn.ReLU())  # keep positives, zero out negatives
            in_size = size
        layers.append(nn.Linear(in_size, 10))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
