"""Tests for the physics_iq module (Physics-IQ full-reference protocol).

These tests exercise the ported metric with tiny synthetic videos and use no
ML dependencies (pure cv2 + numpy). The identical pair must yield near-perfect
IoU / near-zero MSE, while a motion-mismatched pair must yield clearly lower
spatial and spatiotemporal IoU. A sample without a reference must leave every
physics_iq_* field None.
"""

import cv2
import numpy as np
import pytest

from ..conftest import _test_module_basics
from ayase.models import Sample


def _make_square_video(path, *, size=64, n_frames=20, square=16, y=8, step=3):
    """Write a video of a white square sweeping horizontally at a fixed row band.

    The row band [y, y+square) fixes where motion appears; different ``y`` values
    for two videos produce disjoint motion masks.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (size, size))
    assert writer.isOpened(), "cv2.VideoWriter (mp4v) unavailable"
    max_x = size - square
    for i in range(n_frames):
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        x = (2 + step * i) % max_x
        cv2.rectangle(frame, (x, y), (x + square, y + square), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path


# Keep masks at full resolution so the small synthetic squares stay robust.
_CFG = {"downscale_factor": 1}


def _new_module():
    from ayase.modules.physics_iq import PhysicsIQModule
    return PhysicsIQModule(_CFG)


def test_physics_iq_basics():
    from ayase.modules.physics_iq import PhysicsIQModule
    _test_module_basics(PhysicsIQModule, "physics_iq")


def test_physics_iq_identical_pair(tmp_path):
    """Identical generated/real videos -> IoU ~1, MSE ~0, score ~100."""
    vid = _make_square_video(tmp_path / "gen.mp4")
    sample = Sample(path=vid, is_video=True, reference_path=vid)

    m = _new_module()
    result = m.process(sample)
    qm = result.quality_metrics

    assert qm is not None
    assert qm.physics_iq_spatial_iou == pytest.approx(1.0, abs=1e-6)
    assert qm.physics_iq_spatiotemporal_iou == pytest.approx(1.0, abs=1e-6)
    assert qm.physics_iq_weighted_spatial_iou == pytest.approx(1.0, abs=1e-6)
    assert qm.physics_iq_mse == pytest.approx(0.0, abs=1e-6)
    assert qm.physics_iq_score == pytest.approx(100.0, abs=1e-6)
    # Deterministic port actually computed -> backend recorded.
    assert m._backend == "port"


def test_physics_iq_motion_mismatch(tmp_path):
    """Disjoint motion bands -> clearly lower spatial/spatiotemporal IoU, higher MSE."""
    ref = _make_square_video(tmp_path / "real.mp4", y=8)      # top band
    gen = _make_square_video(tmp_path / "gen.mp4", y=40)      # bottom band (disjoint)
    sample = Sample(path=gen, is_video=True, reference_path=ref)

    m = _new_module()
    qm = m.process(sample).quality_metrics

    assert qm is not None
    # All fields populated (metric actually ran).
    for val in (
        qm.physics_iq_spatial_iou,
        qm.physics_iq_spatiotemporal_iou,
        qm.physics_iq_weighted_spatial_iou,
        qm.physics_iq_mse,
        qm.physics_iq_score,
    ):
        assert val is not None

    # Disjoint motion -> IoU collapses far below the identical case.
    assert qm.physics_iq_spatial_iou < 0.5
    assert qm.physics_iq_spatiotemporal_iou < 0.5
    assert qm.physics_iq_mse > 1e-4

    # And clearly worse than an identical pair on the same generator.
    ident = Sample(path=gen, is_video=True, reference_path=gen)
    ident_qm = _new_module().process(ident).quality_metrics
    assert qm.physics_iq_spatial_iou < ident_qm.physics_iq_spatial_iou
    assert qm.physics_iq_spatiotemporal_iou < ident_qm.physics_iq_spatiotemporal_iou
    assert qm.physics_iq_score < ident_qm.physics_iq_score


def test_physics_iq_no_reference(tmp_path):
    """No reference -> every physics_iq_* field stays None."""
    vid = _make_square_video(tmp_path / "gen.mp4")
    sample = Sample(path=vid, is_video=True)  # no reference_path

    m = _new_module()
    result = m.process(sample)

    if result.quality_metrics is not None:
        assert result.quality_metrics.physics_iq_score is None
        assert result.quality_metrics.physics_iq_spatial_iou is None
        assert result.quality_metrics.physics_iq_spatiotemporal_iou is None
        assert result.quality_metrics.physics_iq_weighted_spatial_iou is None
        assert result.quality_metrics.physics_iq_mse is None
    assert m._backend is None  # nothing computed
