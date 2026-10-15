"""Phase 5: the final exam. Score every model on the test set, once.

Run from the project root (after training the PyTorch models):
    python -m src.report
"""

import numpy as np
import torch
from torch import nn

from src.baseline import accuracy, make_models, to_arrays
from src.data import get_dataloaders, get_raw_images, load_datasets
from src.evaluate import (
    confusion_matrix,
    plot_confusion_matrix,
    plot_misclassified,
    predict_all,
    top_confusions,
)
from src.explore import OUTPUT_DIR, save_figure
from src.train import MODEL_BUILDERS, MODELS_DIR, build_model, get_device

NUM_MISCLASSIFIED_TO_SHOW = 20


def load_trained_model(name: str, device: torch.device) -> nn.Module:
    """Rebuild a model and fill it with the weights saved by src.train."""
    path = MODELS_DIR / f"{name}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Train it first: python -m src.train --model {name}"
        )
    model = build_model(name)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device)


def score_sklearn_models() -> dict[str, float]:
    """Retrain the scikit-learn baselines on all 55k images and test them."""
    train, _, test = load_datasets()
    X_train, y_train = to_arrays(train)
    X_test, y_test = to_arrays(test)
    scores = {}
    for name, model in make_models().items():
        model.fit(X_train, y_train)
        scores[name] = accuracy(model, X_test, y_test)
    return scores


def analyse_mistakes(name: str, predictions: np.ndarray, labels: np.ndarray) -> None:
    """Print the most common mistakes and save the diagnostic plots."""
    matrix = confusion_matrix(predictions, labels)
    print(f"\nMost common {name.upper()} mistakes (true → predicted):")
    for true, pred, count in top_confusions(matrix):
        print(f"  {true} → {pred}: {count} times")

    _, _, test = load_datasets()
    wrong = np.flatnonzero(predictions != labels)[:NUM_MISCLASSIFIED_TO_SHOW]
    images = get_raw_images(test).numpy()[wrong]

    save_figure(
        plot_confusion_matrix(matrix, title=f"{name.upper()} confusion matrix (test set)"),
        OUTPUT_DIR / f"{name}_confusion_matrix.png",
    )
    save_figure(
        plot_misclassified(images, labels[wrong], predictions[wrong]),
        OUTPUT_DIR / f"{name}_misclassified.png",
    )
    print(f"\nPlots saved to {OUTPUT_DIR}/")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    device = get_device()
    scores = score_sklearn_models()

    _, _, test_loader = get_dataloaders()
    predictions = {}
    for name in MODEL_BUILDERS:
        preds, labels = predict_all(load_trained_model(name, device), test_loader, device)
        predictions[name] = preds.numpy()
        scores[name.upper()] = float((preds == labels).float().mean())
    labels = labels.numpy()

    print("Test accuracy (10,000 images the models have never seen):\n")
    for name, score in scores.items():
        mistakes = round((1 - score) * len(labels))
        print(f"  {name:<20} {score:.2%}   ({mistakes} wrong)")

    analyse_mistakes("cnn", predictions["cnn"], labels)


if __name__ == "__main__":
    main()
