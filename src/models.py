"""PyTorch model definitions: MLP and CNN."""

import torch
from torch import nn


class MLP(nn.Module):
    """A multi-layer perceptron: one hidden layer of 128 neurons.

    Input: a batch of images, shape (batch, 1, 28, 28).
    Output: 10 raw scores ("logits") per image, one per digit.
    There is no softmax here because CrossEntropyLoss applies it internally.
    """

    def __init__(self, hidden_size: int = 128) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),                  # (batch, 1, 28, 28) -> (batch, 784)
            nn.Linear(28 * 28, hidden_size),
            nn.ReLU(),                     # keep positives, zero out negatives
            nn.Linear(hidden_size, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
