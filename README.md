# Handwritten Digit Recognizer

Recognizes handwritten digits (0–9) from the MNIST dataset. Each model is more
powerful than the one before, and all of them were scored once at the end on
10,000 test images they had never seen.

## Results (test set)

| Model                               | Test accuracy | Wrong (of 10,000) |
|-------------------------------------|---------------|-------------------|
| Logistic regression (scikit-learn)  | 92.5%         | 746               |
| Random forest (scikit-learn)        | 96.8%         | 318               |
| MLP, 1 hidden layer (PyTorch)       | 97.7%         | 230               |
| MLP, 2 hidden layers (PyTorch)      | 97.9%         | 210               |
| **CNN (PyTorch)**                   | **99.0%**     | **98**            |

The PyTorch models trained for 5 epochs with Adam (learning rate 1e-3) on an
Apple Silicon GPU (`mps`). See `PLAN.md` for validation scores and timings.

![CNN confusion matrix](outputs/cnn_confusion_matrix.png)

> The images in `outputs/` are git-ignored. Run `python -m src.report` to generate them.

## What I learned

- **Start with a baseline.** Logistic regression trains in seconds and gave a
  number to beat (92.5%). Every later model had to justify its extra complexity
  against it.
- **More data helps.** Logistic regression scored 89.5% with 10k training
  images and 91.9% with 55k.
- **Architecture matters more than size.** The 2-layer MLP and the CNN have
  almost the same number of weights (235k vs 225k), but the CNN makes less
  than half as many mistakes. Convolutions reuse small filters across the
  image, so they learn edges and curves once and find them anywhere.
- **Compare error rates, not just accuracy.** 97.9% → 99.0% looks small, but it
  means going from 210 mistakes to 98.
- **Keep the test set for the end.** All tuning used a 5k validation split.
  The test set was used once, in `src/report.py`, so its scores are honest.
- **Look at the mistakes.** The CNN's most common confusions are 7→2, 9→8,
  6→5 and 9→4. Many of the misclassified images are ambiguous even to a
  person, which suggests we are close to the limit of what this data allows.

## Project layout

```
src/
  data.py       load MNIST, train/validation/test split, DataLoaders
  explore.py    sample grid and class-balance plots
  baseline.py   scikit-learn logistic regression and random forest
  models.py     PyTorch MLP and CNN
  train.py      reusable training loop (any model in MODEL_BUILDERS)
  evaluate.py   accuracy, confusion matrix, and plotting helpers
  report.py     final test-set evaluation of every model
tests/          pytest tests
```

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.explore                      # look at the data
python -m src.baseline --train-size 55000  # scikit-learn baselines
python -m src.train --model mlp            # also: mlp2, cnn
python -m src.report                       # final test-set scores and plots
pytest                                     # run the tests
```
