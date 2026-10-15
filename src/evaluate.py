"""Evaluation helpers: accuracy, confusion matrix, and plots."""

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from torch import nn
from torch.utils.data import DataLoader


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Return the model's accuracy (0.0-1.0) on every batch in the loader."""
    model.eval()  # switch off training-only behaviour such as dropout
    correct = 0
    total = 0
    with torch.no_grad():  # no gradients needed, so this is faster
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images).argmax(dim=1)  # highest score wins
            correct += (predictions == labels).sum().item()
            total += len(labels)
    return correct / total


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
