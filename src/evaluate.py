"""Evaluation helpers: accuracy, confusion matrix, and plots."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay
from torch import nn
from torch.utils.data import DataLoader


def predict_all(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (predictions, true labels) for every example in the loader."""
    model.eval()  # switch off training-only behaviour such as dropout
    all_predictions, all_labels = [], []
    with torch.no_grad():  # no gradients needed, so this is faster
        for images, labels in loader:
            logits = model(images.to(device))
            all_predictions.append(logits.argmax(dim=1).cpu())  # highest score wins
            all_labels.append(labels)
    return torch.cat(all_predictions), torch.cat(all_labels)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Return the model's accuracy (0.0-1.0) on every batch in the loader."""
    predictions, labels = predict_all(model, loader, device)
    return (predictions == labels).float().mean().item()


def confusion_matrix(
    predictions: np.ndarray, labels: np.ndarray, num_classes: int = 10
) -> np.ndarray:
    """Return a num_classes x num_classes table of counts.

    Row = true digit, column = predicted digit. The diagonal holds the
    correct answers; every other cell is one kind of mistake.
    """
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    np.add.at(matrix, (labels, predictions), 1)  # +1 at each (true, predicted)
    return matrix


def top_confusions(matrix: np.ndarray, n: int = 5) -> list[tuple[int, int, int]]:
    """Return the n most common mistakes as (true digit, predicted digit, count)."""
    mistakes = matrix.copy()
    np.fill_diagonal(mistakes, 0)  # ignore correct answers
    pairs = [
        (true, pred, int(mistakes[true, pred]))
        for true in range(len(mistakes))
        for pred in range(len(mistakes))
        if mistakes[true, pred] > 0
    ]
    return sorted(pairs, key=lambda pair: pair[2], reverse=True)[:n]


def plot_confusion_matrix(matrix: np.ndarray, title: str = "") -> Figure:
    """Draw the confusion matrix as a colour-coded grid of counts."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(matrix, display_labels=range(10)).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_misclassified(
    images: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    cols: int = 5,
) -> Figure:
    """Show wrongly predicted images, titled 'true -> predicted'."""
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.6))
    for ax in np.atleast_1d(axes).flat:
        ax.axis("off")
    for ax, image, label, pred in zip(np.atleast_1d(axes).flat, images, labels, predictions):
        ax.imshow(image, cmap="gray")
        ax.set_title(f"{label} → {pred}", color="crimson")
    fig.suptitle("Misclassified test images (true → predicted)")
    fig.tight_layout()
    return fig


def plot_history(
    history: dict[str, list[float]],
    title: str = "",
    eval_name: str = "val",
) -> Figure:
    """Plot training loss and evaluation accuracy for each epoch."""
    eval_label = "Test" if eval_name == "test" else "Validation"
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(10, 4))

    loss_ax.plot(epochs, history["train_loss"], marker="o")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Training loss")
    loss_ax.set_title("Training loss (lower is better)")

    acc_ax.plot(epochs, [acc * 100 for acc in history["eval_acc"]], marker="o")
    acc_ax.set_xlabel("Epoch")
    acc_ax.set_ylabel(f"{eval_label} accuracy (%)")
    acc_ax.set_title(f"{eval_label} accuracy (higher is better)")

    for ax in (loss_ax, acc_ax):
        ax.set_xticks(list(epochs))
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    return fig
