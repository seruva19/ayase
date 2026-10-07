"""Tests for pose_fidelity (AKD/MKR FOMM + MPJPE/PCK Ginosar on MediaPipe pose)."""

import numpy as np
import pytest

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics, Sample


def _seq(rng, t=10, shift=0.0):
    return rng.rand(t, 33, 4).astype(np.float32) * 0.5 + shift


def test_pose_fidelity_basics():
    from ayase.modules.pose_fidelity import PoseFidelityModule
    _test_module_basics(PoseFidelityModule, "pose_fidelity")


def test_pose_fidelity_provenance():
    from ayase.modules.pose_fidelity import PoseFidelityModule
    meta = PoseFidelityModule.get_metadata()
    for f in ("akd", "mkr", "mpjpe", "pck"):
        assert meta["provenance"][f] == "adapted"
    assert "pose-evaluation" in meta["sources"]["akd"]


def test_pose_fidelity_no_backend(tmp_path):
    from ayase.modules.pose_fidelity import PoseFidelityModule
    m = PoseFidelityModule()
    sample = Sample(path=tmp_path / "v.mp4", is_video=True)
    sample.quality_metrics = QualityMetrics()
    assert m.process(sample).quality_metrics.akd is None


def test_pose_fidelity_no_reference_skips(tmp_path):
    from ayase.modules.pose_fidelity import PoseFidelityModule
    m = PoseFidelityModule()
    m._backend = "mediapipe"
    m._pose = object()
    sample = Sample(path=tmp_path / "v.mp4", is_video=True)
    sample.quality_metrics = QualityMetrics()
    assert m.process(sample).quality_metrics.akd is None


def test_pose_fidelity_identical(tmp_path, monkeypatch):
    import ayase.modules.pose_fidelity as pf

    m = pf.PoseFidelityModule()
    m._backend = "mediapipe"
    m._pose = object()
    rng = np.random.RandomState(0)
    seq = _seq(rng)
    seq[:, :, 3] = 1.0  # full visibility
    monkeypatch.setattr("ayase.modules._mp_seq.body_pose_seq",
                        lambda p, **kw: (seq, np.arange(len(seq))))
    ref = tmp_path / "r.mp4"
    ref.write_bytes(b"x")
    sample = Sample(path=tmp_path / "v.mp4", is_video=True, reference_path=ref)
    sample.quality_metrics = QualityMetrics()
    out = m.process(sample)
    assert out.quality_metrics.akd == pytest.approx(0.0)
    assert out.quality_metrics.mkr == pytest.approx(0.0)
    assert out.quality_metrics.mpjpe == pytest.approx(0.0)
    assert out.quality_metrics.pck == pytest.approx(1.0)


def test_pose_fidelity_missing_keypoints(tmp_path, monkeypatch):
    import ayase.modules.pose_fidelity as pf

    m = pf.PoseFidelityModule()
    m._backend = "mediapipe"
    m._pose = object()
    rng = np.random.RandomState(0)
    src = _seq(rng); src[:, :, 3] = 1.0
    gen = src.copy(); gen[:, :, 3] = 0.0  # all candidate joints invisible
    seqs = {"v.mp4": (gen, np.arange(10)), "r.mp4": (src, np.arange(10))}
    monkeypatch.setattr("ayase.modules._mp_seq.body_pose_seq",
                        lambda p, **kw: seqs[p.name])
    ref = tmp_path / "r.mp4"
    ref.write_bytes(b"x")
    sample = Sample(path=tmp_path / "v.mp4", is_video=True, reference_path=ref)
    sample.quality_metrics = QualityMetrics()
    out = m.process(sample)
    assert out.quality_metrics.mkr == pytest.approx(1.0)
    assert out.quality_metrics.pck == pytest.approx(0.0)
