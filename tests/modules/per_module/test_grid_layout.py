"""Tests for the grid_layout module.

Covers module basics and the three canonical cases: a 2x2 colour collage
(high score + detected layout), a natural gradient (low score), and a
letterboxed frame (low score — black bars are not internal dividers). No ML
dependencies are required.
"""

from pathlib import Path

import cv2
import numpy as np

from ..conftest import _test_module_basics
from ayase.models import Sample
from ayase.modules.grid_layout import GridLayoutModule


def _save(tmp_dir: Path, name: str, img: np.ndarray) -> Path:
    path = tmp_dir / name
    cv2.imwrite(str(path), img)
    return path


def _collage_2x2(size: int = 256) -> np.ndarray:
    """Four distinctly coloured quadrants (BGR)."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    h = size // 2
    img[:h, :h] = (0, 0, 255)     # top-left  red
    img[:h, h:] = (0, 255, 0)     # top-right green
    img[h:, :h] = (255, 0, 0)     # bot-left  blue
    img[h:, h:] = (0, 255, 255)   # bot-right yellow
    return img


def _gradient(size: int = 256) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for x in range(size):
        img[:, x, :] = x
    noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
    return cv2.add(img, noise)


def _letterboxed(size: int = 256, bar: int = 40) -> np.ndarray:
    img = _gradient(size)
    img[:bar, :, :] = 0        # top black bar
    img[size - bar:, :, :] = 0  # bottom black bar
    return img


def test_grid_layout_basics():
    _test_module_basics(GridLayoutModule, "grid_layout")


def test_collage_scores_high(tmp_dir):
    m = GridLayoutModule()
    sample = Sample(path=_save(tmp_dir, "collage.png", _collage_2x2()), is_video=False)
    out = m.process(sample)
    assert out.quality_metrics is not None
    score = out.quality_metrics.grid_layout_score
    assert score is not None and score > 0.6, f"expected high grid score, got {score}"
    # Layout stored in metadata for a high-scoring grid.
    meta = getattr(out, "metadata", {})
    assert meta.get("grid_layout"), "expected a detected layout in metadata"


def test_gradient_scores_low(tmp_dir):
    m = GridLayoutModule()
    sample = Sample(path=_save(tmp_dir, "gradient.png", _gradient()), is_video=False)
    out = m.process(sample)
    score = out.quality_metrics.grid_layout_score
    assert score is not None and score < 0.3, f"expected low grid score, got {score}"
    assert getattr(out, "metadata", {}).get("grid_layout") is None


def test_letterbox_is_not_a_grid(tmp_dir):
    """Black letterbox bars must not be detected as internal grid dividers."""
    m = GridLayoutModule()
    sample = Sample(path=_save(tmp_dir, "letterbox.png", _letterboxed()), is_video=False)
    out = m.process(sample)
    score = out.quality_metrics.grid_layout_score
    assert score is not None and score < 0.3, f"letterbox misread as grid, got {score}"
    assert getattr(out, "metadata", {}).get("grid_layout") is None


def test_backend_is_algorithmic():
    assert GridLayoutModule()._backend == "algorithmic"
