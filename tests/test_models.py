"""Tests for src/models.py."""

import torch

from src.models import MLP


def test_mlp_output_shape():
    model = MLP()
    dummy_images = torch.randn(8, 1, 28, 28)  # a fake batch of 8 images
    assert model(dummy_images).shape == (8, 10)
