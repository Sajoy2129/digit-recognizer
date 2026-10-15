"""Tests for src/models.py."""

import torch

from src.models import CNN, MLP


def test_cnn_output_shape():
    model = CNN()
    assert model(torch.randn(8, 1, 28, 28)).shape == (8, 10)


def test_cnn_features_shrink_to_64x5x5():
    features = CNN().features(torch.randn(2, 1, 28, 28))
    assert features.shape == (2, 64, 5, 5)


def test_mlp_output_shape():
    model = MLP()
    dummy_images = torch.randn(8, 1, 28, 28)  # a fake batch of 8 images
    assert model(dummy_images).shape == (8, 10)


def test_two_hidden_layer_mlp_output_shape():
    model = MLP(hidden_sizes=(256, 128))
    assert model(torch.randn(8, 1, 28, 28)).shape == (8, 10)


def test_two_hidden_layer_mlp_has_three_linear_layers():
    model = MLP(hidden_sizes=(256, 128))
    linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    sizes = [(layer.in_features, layer.out_features) for layer in linears]
    assert sizes == [(784, 256), (256, 128), (128, 10)]
