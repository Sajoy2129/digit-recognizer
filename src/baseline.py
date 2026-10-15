"""scikit-learn baseline models (logistic regression, random forest) for MNIST.

Run from the project root:
    python -m src.baseline
    python -m src.baseline --train-size 55000   # use the full training split
"""

import argparse
import time

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from torch.utils.data import Dataset

from src.data import SEED, get_labels, get_raw_images, load_datasets

DEFAULT_TRAIN_SIZE = 10_000


def to_arrays(dataset: Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Turn an MNIST dataset into (X, y) NumPy arrays for scikit-learn.

    X has shape (N, 784): each 28x28 image flattened into one row,
    with pixels scaled from 0-255 to 0.0-1.0.
    y has shape (N,): the digit labels.
    """
    images = get_raw_images(dataset).numpy()
    X = images.reshape(len(images), -1).astype(np.float32) / 255.0
    y = get_labels(dataset).numpy()
    return X, y


def make_models() -> dict[str, ClassifierMixin]:
    """Return the baseline models to compare, keyed by a readable name."""
    return {
        # max_iter: how many optimization steps the solver may take.
        "Logistic regression": LogisticRegression(max_iter=1000),
        # n_jobs=-1 uses every CPU core to build the trees in parallel.
        "Random forest": RandomForestClassifier(
            n_estimators=100, n_jobs=-1, random_state=SEED
        ),
    }


def accuracy(model: ClassifierMixin, X: np.ndarray, y: np.ndarray) -> float:
    """Return the fraction of examples the model labels correctly."""
    return float((model.predict(X) == y).mean())


def train_and_score(
    model: ClassifierMixin,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[float, float]:
    """Fit the model, then return (validation accuracy, training seconds)."""
    start = time.perf_counter()
    model.fit(X_train, y_train)
    seconds = time.perf_counter() - start
    return accuracy(model, X_val, y_val), seconds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train scikit-learn baselines.")
    parser.add_argument(
        "--train-size",
        type=int,
        default=DEFAULT_TRAIN_SIZE,
        help="number of training images to use (max 55000)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train, val, _ = load_datasets()  # the test set waits until Phase 5

    X_train, y_train = to_arrays(train)
    X_val, y_val = to_arrays(val)
    # The train split is already shuffled, so the first N rows are a fair sample.
    X_train, y_train = X_train[: args.train_size], y_train[: args.train_size]
    print(f"Training on {len(X_train):,} images, "
          f"validating on {len(X_val):,}. Features per image: {X_train.shape[1]}\n")

    for name, model in make_models().items():
        val_acc, seconds = train_and_score(model, X_train, y_train, X_val, y_val)
        print(f"{name:<20} val accuracy {val_acc:.2%}   train time {seconds:.1f}s")


if __name__ == "__main__":
    main()
