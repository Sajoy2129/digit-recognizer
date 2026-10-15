"""Training and evaluation loop shared by the PyTorch models.

Run from the project root:
    python -m src.train --model mlp
    python -m src.train --model mlp --epochs 10 --lr 0.0005
"""

import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data import SEED, get_dataloaders
from src.evaluate import evaluate, plot_history
from src.explore import OUTPUT_DIR, save_figure
from src.models import MLP

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Add new architectures here (Phase 4 adds "cnn").
MODEL_CLASSES = {"mlp": MLP}


def get_device() -> torch.device:
    """Use the Apple Silicon GPU ("mps") if available, otherwise the CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(name: str) -> nn.Module:
    """Create a fresh, untrained model by name."""
    return MODEL_CLASSES[name]()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one pass over the data and return the average loss."""
    model.train()  # switch on training-only behaviour such as dropout
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()            # 1. clear gradients from the last step
        logits = model(images)           # 2. forward pass: make predictions
        loss = loss_fn(logits, labels)   # 3. measure how wrong they are
        loss.backward()                  # 4. work out how to adjust each weight
        optimizer.step()                 # 5. adjust the weights a little

        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> dict[str, list[float]]:
    """Train the model and return its per-epoch history of loss and accuracy."""
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch}/{epochs}  "
              f"train loss {train_loss:.4f}  val accuracy {val_acc:.2%}")
    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PyTorch model on MNIST.")
    parser.add_argument("--model", choices=MODEL_CLASSES, default="mlp")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Seed first, so the starting weights and the shuffle order repeat each run.
    torch.manual_seed(SEED)
    device = get_device()
    print(f"Model: {args.model}  Device: {device}\n")

    train_loader, val_loader, _ = get_dataloaders()  # test set waits for Phase 5
    model = build_model(args.model)

    start = time.perf_counter()
    history = fit(model, train_loader, val_loader, args.epochs, args.lr, device)
    print(f"\nTraining time: {time.perf_counter() - start:.1f}s")

    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    weights_path = MODELS_DIR / f"{args.model}.pt"
    curves_path = OUTPUT_DIR / f"{args.model}_curves.png"
    torch.save(model.state_dict(), weights_path)
    save_figure(plot_history(history, title=args.model.upper()), curves_path)
    print(f"Saved weights to {weights_path}")
    print(f"Saved curves to {curves_path}")


if __name__ == "__main__":
    main()
