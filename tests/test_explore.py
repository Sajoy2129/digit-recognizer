"""Tests for src/explore.py."""

import torch

from src.explore import format_image


def test_format_image_has_one_line_per_row():
    image = torch.zeros(28, 28, dtype=torch.uint8)
    lines = format_image(image).splitlines()
    assert len(lines) == 28


def test_format_image_shows_pixel_values_and_dots():
    image = torch.zeros(28, 28, dtype=torch.uint8)
    image[0, 0] = 255
    first_line = format_image(image).splitlines()[0]
    assert first_line.startswith("255")
    assert "  ." in first_line
