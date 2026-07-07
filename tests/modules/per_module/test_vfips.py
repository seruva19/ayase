"""Tests for vfips module."""

import numpy as np
import pytest

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics, Sample


def test_vfips_basics():
    from ayase.modules.vfips import VFIPSModule
    _test_module_basics(VFIPSModule, "vfips")


def test_vfips_no_reference(image_sample):
    from ayase.modules.vfips import VFIPSModule
    m = VFIPSModule()
    result = m.process(image_sample)
    assert result is image_sample


def _real_backend_or_skip():
    """Instantiate the module with its real trained backend, or skip."""
    from ayase.modules.vfips import VFIPSModule

    m = VFIPSModule()
    try:
        m.setup()
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"VFIPS setup raised: {e}")
    if m._backend != "real":
        pytest.skip("VFIPS real backend/weights unavailable")
    return m


def _write_video(path, n=16, size=160, noise=0.0, seed=0):
    import cv2

    rng = np.random.default_rng(seed)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (size, size))
    base = []
    for i in range(n):
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        cx = size // 2 + int(30 * np.sin(i * 0.3))
        cy = size // 2 + int(30 * np.cos(i * 0.3))
        cv2.circle(frame, (cx, cy), 24, (0, 200, 80), -1)
        for x in range(size):
            frame[:, x, 0] = min(255, x + i)
        base.append(frame)
    for frame in base:
        if noise > 0:
            n_img = (rng.standard_normal(frame.shape) * (noise * 255)).astype(np.int16)
            frame = np.clip(frame.astype(np.int16) + n_img, 0, 255).astype(np.uint8)
        writer.write(frame)
    writer.release()


@pytest.mark.full
def test_vfips_real_reference(tmp_dir):
    """Weights-guarded: the trained network yields a real vfips_score and a
    distorted clip scores as more distant than an identical reference."""
    m = _real_backend_or_skip()

    ref = tmp_dir / "ref.mp4"
    dist = tmp_dir / "dist.mp4"
    _write_video(ref, noise=0.0, seed=1)
    _write_video(dist, noise=0.15, seed=1)

    # Identical reference -> self-distance.
    same = Sample(path=ref, is_video=True, reference_path=ref)
    same = m.process(same)
    assert same.quality_metrics is not None
    assert same.quality_metrics.vfips_score is not None
    self_score = float(same.quality_metrics.vfips_score)

    # Distorted vs reference -> larger perceptual distance.
    diff = Sample(path=dist, is_video=True, reference_path=ref)
    diff = m.process(diff)
    assert diff.quality_metrics.vfips_score is not None
    dist_score = float(diff.quality_metrics.vfips_score)

    assert np.isfinite(self_score) and np.isfinite(dist_score)
    assert dist_score > self_score
