"""Training and evaluation loop shared by the PyTorch models.

Run from the project root:
    python -m src.train --model mlp
    python -m src.train --model mlp --epochs 10 --lr 0.0005
"""

import argparse
import time
from functools import partial
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data import SEED, get_dataloaders
from src.evaluate import evaluate, plot_history
from src.explore import OUTPUT_DIR, save_figure
from src.models import MLP

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Each entry creates a fresh model. partial() pre-fills constructor arguments.
# Add new architectures here (Phase 4 adds "cnn").
MODEL_BUILDERS = {
    "mlp": MLP,                                     # 1 hidden layer (128)
    "mlp2": partial(MLP, hidden_sizes=(256, 128)),  # 2 hidden layers
}


def get_device() -> torch.device:
    """Use the Apple Silicon GPU ("mps") if available, otherwise the CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(name: str) -> nn.Module:
    """Create a fresh, untrained model by name."""
    return MODEL_BUILDERS[name]()


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
    eval_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    eval_name: str = "val",
) -> dict[str, list[float]]:
    """Train the model and return its per-epoch history of loss and accuracy.

    eval_loader is checked after every epoch; eval_name labels it in the output.
    """
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "eval_acc": []}

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        eval_acc = evaluate(model, eval_loader, device)
        history["train_loss"].append(train_loss)
        history["eval_acc"].append(eval_acc)
        print(f"Epoch {epoch}/{epochs}  "
              f"train loss {train_loss:.4f}  {eval_name} accuracy {eval_acc:.2%}")
    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PyTorch model on MNIST.")
    parser.add_argument("--model", choices=MODEL_BUILDERS, default="mlp")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--eval-split",
        choices=["val", "test"],
        default="val",
        help="data to check after each epoch; keep 'val' while tuning, "
             "because repeatedly checking the test set makes it less honest",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Seed first, so the starting weights and the shuffle order repeat each run.
    torch.manual_seed(SEED)
    device = get_device()
    print(f"Model: {args.model}  Device: {device}  "
          f"Evaluating on: {args.eval_split}\n")

    train_loader, val_loader, test_loader = get_dataloaders()
    eval_loader = test_loader if args.eval_split == "test" else val_loader
    model = build_model(args.model)

    start = time.perf_counter()
    history = fit(model, train_loader, eval_loader, args.epochs, args.lr,
                  device, eval_name=args.eval_split)
    print(f"\nTraining time: {time.perf_counter() - start:.1f}s")

    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    weights_path = MODELS_DIR / f"{args.model}.pt"
    curves_path = OUTPUT_DIR / f"{args.model}_curves.png"
    torch.save(model.state_dict(), weights_path)
    fig = plot_history(history, title=args.model.upper(), eval_name=args.eval_split)
    save_figure(fig, curves_path)
    print(f"Saved weights to {weights_path}")
    print(f"Saved curves to {curves_path}")


if __name__ == "__main__":
    main()
