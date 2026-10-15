"""Tests for src/baseline.py."""

import numpy as np
import pytest

from src.baseline import accuracy, make_models, to_arrays
from src.data import load_datasets


@pytest.fixture(scope="module")
def val_arrays():
    _, val, _ = load_datasets()
    return to_arrays(val)


def test_to_arrays_flattens_images(val_arrays):
    X, y = val_arrays
    assert X.shape == (len(y), 784)
    assert y.shape == (len(X),)


def test_to_arrays_scales_pixels_to_0_1(val_arrays):
    X, _ = val_arrays
    assert X.min() == 0.0
    assert X.max() == 1.0


@pytest.mark.parametrize("name", list(make_models()))
def test_predictions_have_right_shape_and_are_digits(val_arrays, name):
    X, y = val_arrays
    X_small, y_small = X[:500], y[:500]  # tiny subset keeps the test fast
    model = make_models()[name]
    model.fit(X_small, y_small)

    predictions = model.predict(X_small)
    assert predictions.shape == (500,)
    assert set(np.unique(predictions)) <= set(range(10))


def test_accuracy_counts_correct_fraction():
    class AlwaysSeven:
        def predict(self, X):
            return np.full(len(X), 7)

    y = np.array([7, 7, 1, 2])
    assert accuracy(AlwaysSeven(), np.zeros((4, 784)), y) == 0.5
