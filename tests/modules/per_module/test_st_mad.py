"""Tests for st_mad module."""

import cv2
import numpy as np

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_st_mad_basics():
    from ayase.modules.st_mad import STMADModule
    _test_module_basics(STMADModule, "st_mad")


def test_st_mad_no_reference(image_sample):
    from ayase.modules.st_mad import STMADModule
    m = STMADModule()
    result = m.process(image_sample)
    assert result is image_sample


def _write_video(path, frames):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()


def test_st_mad_real_path(tmp_dir):
    """Deterministic classical port: identical clip ~ baseline, noise raises score."""
    from ayase.modules.st_mad import STMADModule

    rng = np.random.default_rng(0)
    h = w = 64
    n = 44
    base = []
    for i in range(n):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # smooth horizontal ramp (low masking -> detection stage active)
        for x in range(w):
            frame[:, x, :] = min(255, x * 3)
        # moving bright box -> real inter-frame motion for the flow weights
        cx = 6 + i
        frame[24:40, max(0, cx - 6):min(w, cx + 6), :] = 240
        base.append(frame)

    ref_path = tmp_dir / "ref.mp4"
    low_path = tmp_dir / "dist_low.mp4"
    high_path = tmp_dir / "dist_high.mp4"

    _write_video(ref_path, base)
    _write_video(low_path, base)  # same content -> ~baseline after identical encode
    high = [np.clip(f.astype(np.float64) + rng.normal(0, 28, f.shape), 0, 255).astype(np.uint8)
            for f in base]
    _write_video(high_path, high)

    m = STMADModule()
    m.setup()

    s_low = m.compute_reference_score(low_path, ref_path)
    s_high = m.compute_reference_score(high_path, ref_path)

    assert s_low is not None and s_high is not None
    assert np.isfinite(s_low) and np.isfinite(s_high)
    # More distortion => higher ST-MAD (lower is better).
    assert s_high > s_low

    # Still images are not defined for a temporal metric.
    assert m.compute_reference_score(tmp_dir / "x.png", ref_path) is None


def test_st_mad_stores_metric_field():
    """metric_field wiring stores the score into QualityMetrics.st_mad."""
    qm = QualityMetrics()
    assert qm.st_mad is None
    from ayase.modules.st_mad import STMADModule
    assert STMADModule.metric_field == "st_mad"
