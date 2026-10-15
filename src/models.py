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


class CNN(nn.Module):
    """A convolutional neural network: two conv blocks, then a classifier.

    Shapes for one image (channels x height x width):
        1x28x28 -> conv -> 32x26x26 -> pool -> 32x13x13
                -> conv -> 64x11x11 -> pool -> 64x5x5 -> 1600 numbers
    """

    def __init__(self) -> None:
        super().__init__()
        # Feature extractor: learns local patterns (edges, then curves/loops).
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),   # 32 learned 3x3 filters
            nn.ReLU(),
            nn.MaxPool2d(2),                   # keep the strongest of each 2x2
            nn.Conv2d(32, 64, kernel_size=3),  # 64 filters combining those
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Classifier: turns the detected patterns into 10 digit scores.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Dropout(0.25),  # randomly silence 25% of neurons while training
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
