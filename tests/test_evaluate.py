"""Tests for the confusion-matrix and plotting helpers in src/evaluate.py."""

import numpy as np

from src.evaluate import confusion_matrix, plot_misclassified, top_confusions


def test_confusion_matrix_counts_true_vs_predicted():
    labels = np.array([0, 0, 1, 4, 4, 4])
    predictions = np.array([0, 1, 1, 9, 9, 4])
    matrix = confusion_matrix(predictions, labels)

    assert matrix.shape == (10, 10)
    assert matrix.sum() == len(labels)   # every example counted once
    assert matrix[0, 0] == 1             # one 0 predicted correctly
    assert matrix[0, 1] == 1             # one 0 mistaken for a 1
    assert matrix[4, 9] == 2             # two 4s mistaken for 9s
    assert np.trace(matrix) == 3         # diagonal = correct answers


def test_top_confusions_ignores_correct_answers_and_sorts():
    labels = np.array([4, 4, 4, 7, 3, 3])
    predictions = np.array([9, 9, 4, 1, 3, 3])
    matrix = confusion_matrix(predictions, labels)

    assert top_confusions(matrix) == [(4, 9, 2), (7, 1, 1)]


def test_plot_misclassified_draws_one_panel_per_image():
    images = np.zeros((7, 28, 28))
    fig = plot_misclassified(images, np.arange(7), np.arange(7) + 1, cols=5)
    titled = [ax for ax in fig.axes if ax.get_title()]
    assert len(titled) == 7
