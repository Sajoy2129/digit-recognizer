# Handwritten Digit Recognizer — Plan

Goal: recognize handwritten digits (0–9) from the MNIST dataset. We build three
models, each one more powerful than the last, and compare them:

1. **Baseline** — scikit-learn logistic regression (a simple, classic model)
2. **Simple neural network** — a PyTorch multi-layer perceptron (MLP)
3. **Convolutional neural network (CNN)** — a PyTorch model built for images

> **Why a baseline first?** It gives us a number to beat. If a fancy neural
> network can't beat a simple model, something is wrong.

**MNIST in one line:** 70,000 grayscale images (28×28 pixels) of handwritten
digits — 60,000 for training, 10,000 for testing.

---

## Phase 0 — Project setup

- [x] Create a virtual environment (`python3 -m venv .venv`)
- [x] Install dependencies (torch, torchvision, scikit-learn, matplotlib)
- [x] Add a `.gitignore` (ignore `.venv/`, `data/`, `__pycache__/`, `*.pt`)
- [x] Commit `requirements.txt`, `.gitignore`, and `PLAN.md`
- [x] Create the folder layout:
  ```
  digit-recognizer/
  ├── src/
  │   ├── data.py          # load MNIST, build DataLoaders
  │   ├── baseline.py      # scikit-learn model
  │   ├── models.py        # MLP and CNN classes
  │   ├── train.py         # training + evaluation loop
  │   ├── evaluate.py      # accuracy, confusion matrix, plots
  │   └── explore.py       # sample grid + class balance plots
  ├── tests/               # pytest tests
  ├── data/                # MNIST downloads here (git-ignored)
  ├── outputs/             # generated plots (git-ignored)
  ├── explore.ipynb        # interactive tour of the data
  └── models/              # saved model weights (git-ignored)
  ```
- [x] Add `pytest` to the venv and `requirements.txt`

## Phase 1 — Load and explore the data

- [x] `src/data.py`: download MNIST with `torchvision.datasets.MNIST`
- [x] Convert images to tensors and normalize pixel values
      (mean 0.1307, std 0.3081, which are standard for MNIST)
- [x] Split the 60k training set into **train (55k) / validation (5k)**
  - *Why?* We tune using validation and only look at the test set at the very end
- [x] Build `DataLoader`s (batches of 64, shuffle training data)
- [x] Plot a grid of sample images with their labels
- [x] Check the class balance (how many of each digit?)
- [x] Test: a batch has shape `(64, 1, 28, 28)` and labels are in 0–9
- [x] Commit: "Add MNIST data loading and exploration"

## Phase 2 — scikit-learn baseline

- [x] `src/baseline.py`: flatten each 28×28 image into a 784-number vector
- [x] Scale pixels to 0–1
- [x] Train `LogisticRegression` (start with a 10k-sample subset for speed)
- [x] Measure validation accuracy (**expect about 92%**)
- [x] Optional: try `RandomForestClassifier` (about 97%) and compare
- [x] Record results in the results table below
- [x] Test: the model's predictions have the right shape and are in 0–9
- [x] Commit: "Add scikit-learn baseline"

## Phase 3 — Simple neural network (MLP) in PyTorch

- [x] `src/models.py`: define `MLP(nn.Module)`
  - `Flatten → Linear(784, 128) → ReLU → Linear(128, 10)`
- [x] `src/train.py`: write a reusable training loop
  - forward pass → loss (`CrossEntropyLoss`) → `backward()` → optimizer step
  - Optimizer: `Adam`, learning rate 1e-3
  - Use the GPU if available (`mps` on Apple Silicon, otherwise `cpu`)
- [x] Write an `evaluate()` function that computes accuracy with `torch.no_grad()`
- [x] Train for about 5 epochs and print the loss and validation accuracy each epoch
- [x] Plot training loss and validation accuracy curves
- [x] Set a random seed (`torch.manual_seed`) so results are reproducible
- [x] Target: **about 97–98%** validation accuracy
- [x] Save weights with `torch.save(model.state_dict(), "models/mlp.pt")`
- [x] Test: the model's output shape is `(batch, 10)` for a dummy input
- [x] Test: one training step lowers the loss on a tiny batch (sanity check)
- [x] Commit: "Add MLP model and training loop"

## Phase 4 — Convolutional neural network (CNN)

> **Why a CNN?** An MLP treats pixels as a flat list and ignores which pixels
> are neighbors. A CNN slides small filters across the image to find
> local patterns like edges and curves, and that works much better for images.

- [x] `src/models.py`: define `CNN(nn.Module)`
  - `Conv2d(1, 32, 3) → ReLU → MaxPool2d(2)`
  - `Conv2d(32, 64, 3) → ReLU → MaxPool2d(2)`
  - `Flatten → Linear(64*5*5, 128) → ReLU → Dropout(0.25) → Linear(128, 10)`
- [x] Reuse the same `train.py` loop (no changes needed, which is the benefit of
      keeping functions small and general)
- [x] Train for about 5–10 epochs
- [x] Target: **about 99%** validation accuracy
- [ ] Optional: add data augmentation (small random rotations/shifts)
- [x] Save weights to `models/cnn.pt`
- [x] Test: the CNN's output shape is `(batch, 10)`
- [x] Commit: "Add CNN model"

## Phase 5 — Evaluate and compare

- [x] Evaluate **all three models on the test set**, once, at the end
- [x] Fill in the results table
- [x] Plot a confusion matrix for the CNN (which digits get mixed up?)
- [x] Show 10–20 misclassified images and look for patterns
- [x] Write a short summary of what you learned in `README.md`
- [x] Commit: "Add evaluation and results"

## Phase 6 — Stretch goals (optional)

- [ ] Predict a digit from your own image file (`predict.py path/to/img.png`)
- [ ] Build a small drawing app (e.g., Gradio) to draw a digit and see the prediction
- [ ] Try a learning-rate scheduler or batch normalization
- [ ] Track experiments (e.g., TensorBoard)

---

## Results

| Model                     | Val accuracy | Test accuracy | Train time | Notes |
|---------------------------|--------------|---------------|------------|-------|
| Logistic regression       | 89.5% / 91.9% | 92.5% (55k)  | 0.7s / 4.1s | 10k / 55k train images |
| Random forest (bonus)     | 94.2% / 96.7% | 96.8% (55k)  | 0.3s / 2.3s | 10k / 55k train images |
| MLP (PyTorch)             | 97.1%        | 97.7%         | 18.4s (mps) | 5 epochs, Adam lr 1e-3 |
| MLP, 2 hidden layers      | 97.9%        | 97.9%         | 17.9s (mps) | 256→128, 235k params |
| CNN (PyTorch)             | 98.8%        | **99.0%**     | 33.1s (mps) | 5 epochs, 225k params |

## How to run (fill in as we build)

```bash
source .venv/bin/activate
python -m src.baseline                     # 10k-image subset (fast)
python -m src.baseline --train-size 55000  # full training split
python -m src.train --model mlp
python -m src.train --model mlp2
python -m src.train --model cnn
python -m src.report                       # final test-set evaluation
pytest
```
