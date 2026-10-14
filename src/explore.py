"""Explore MNIST: show sample digits, one image as numbers, and class balance.

Run from the project root:
    python -m src.explore
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

from src.data import count_labels, get_raw_image, load_datasets

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def format_image(image: torch.Tensor) -> str:
    """Return a 28x28 image as 28 lines of numbers, each 3 characters wide.

    Background zeros are shown as '  .' so the digit's shape stands out.
    """
    lines = []
    for row in image.tolist():
        cells = [f"{pixel:3d}" if pixel > 0 else "  ." for pixel in row]
        lines.append(" ".join(cells))
    return "\n".join(lines)


def plot_sample_grid(dataset: Dataset, path: Path, rows: int = 4, cols: int = 5) -> None:
    """Save a grid of the first rows*cols images with their labels."""
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.4))
    for i, ax in enumerate(axes.flat):
        image, label = dataset[i]
        ax.imshow(image.squeeze(), cmap="gray")  # (1, 28, 28) -> (28, 28)
        ax.set_title(str(label))
        ax.axis("off")
    fig.suptitle("MNIST training samples")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_class_balance(counts: list[int], path: Path) -> None:
    """Save a bar chart of how many examples there are per digit."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(10), counts, color="steelblue")
    ax.set_xticks(range(10))
    ax.set_xlabel("Digit")
    ax.set_ylabel("Number of training images")
    ax.set_title("Class balance (training split)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    train, val, test = load_datasets()
    print(f"Train: {len(train):,}  Validation: {len(val):,}  Test: {len(test):,}")

    counts = count_labels(train)
    print("\nImages per digit (training split):")
    for digit, count in enumerate(counts):
        share = count / len(train)
        print(f"  {digit}: {count:>5,}  ({share:.1%})")

    raw = get_raw_image(train, 0)
    normalized, label = train[0]
    print(f"\nFirst training image (label {label}), "
          f"shape {tuple(raw.shape)}, pixel values 0-255:\n")
    print(format_image(raw))
    print(f"\nSame image after normalization: values from "
          f"{normalized.min():.2f} to {normalized.max():.2f}")

    plot_sample_grid(train, OUTPUT_DIR / "sample_digits.png")
    plot_class_balance(counts, OUTPUT_DIR / "class_balance.png")
    print(f"\nPlots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
