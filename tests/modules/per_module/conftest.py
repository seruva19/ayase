"""Per-module test fixtures with session-scoped synthetic media.

Creates synthetic image/video ONCE per test session, shared across
all 312+ per-module test files. Much faster than per-test creation.
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from ayase.models import Sample


@pytest.fixture(scope="session")
def _shared_tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture(scope="session")
def _shared_synthetic_image(_shared_tmp_dir):
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    for x in range(256):
        img[:, x, :] = x
    noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    path = _shared_tmp_dir / "test_image.png"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture(scope="session")
def _shared_synthetic_video(_shared_tmp_dir):
    path = _shared_tmp_dir / "test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (256, 256))
    for i in range(64):
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
        cx = 128 + int(50 * np.sin(i * 0.2))
        cy = 128 + int(50 * np.cos(i * 0.2))
        cv2.circle(frame, (cx, cy), 30, (0, 255, 0), -1)
        for x in range(256):
            frame[:, x, 0] = min(255, x + i)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def image_sample(_shared_synthetic_image):
    """Fresh Sample wrapping the shared synthetic image."""
    return Sample(path=_shared_synthetic_image, is_video=False)


@pytest.fixture
def video_sample(_shared_synthetic_video):
    """Fresh Sample wrapping the shared synthetic video."""
    return Sample(path=_shared_synthetic_video, is_video=True)
