"""Tests for the lmd module (Chen 2018 LMD/F-LMD on MediaPipe face mesh)."""

import numpy as np
import pytest

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics, Sample


def test_lmd_basics():
    from ayase.modules.lmd import LMDModule
    _test_module_basics(LMDModule, "lmd")


def test_lmd_provenance():
    from ayase.modules.lmd import LMDModule
    meta = LMDModule.get_metadata()
    assert meta["provenance"]["lmd"] == "adapted"
    assert "Chen" in meta["sources"]["lmd"] or "1803.10404" in meta["sources"]["lmd"]


def test_lmd_no_backend(tmp_path):
    from ayase.modules.lmd import LMDModule
    m = LMDModule()
    sample = Sample(path=tmp_path / "v.mp4", is_video=True)
    sample.quality_metrics = QualityMetrics()
    assert m.process(sample).quality_metrics.lmd is None


def test_lmd_no_reference_skips(tmp_path):
    from ayase.modules.lmd import LMDModule
    m = LMDModule()
    m._backend = "mediapipe"
    m._mesh = object()
    m._lips = {0, 1}
    sample = Sample(path=tmp_path / "v.mp4", is_video=True)
    sample.quality_metrics = QualityMetrics()
    assert m.process(sample).quality_metrics.lmd is None


def test_lmd_identical_meshes_zero(tmp_path, monkeypatch):
    import ayase.modules.lmd as lmod

    m = lmod.LMDModule()
    m._backend = "mediapipe"
    m._mesh = object()
    m._lips = {0, 1, 2}
    seq = np.random.RandomState(0).rand(10, 468, 3).astype(np.float32)
    monkeypatch.setattr("ayase.modules._mp_seq.face_mesh_seq",
                        lambda p, **kw: (seq, np.arange(10)))
    ref = tmp_path / "r.mp4"
    ref.write_bytes(b"x")
    sample = Sample(path=tmp_path / "v.mp4", is_video=True, reference_path=ref)
    sample.quality_metrics = QualityMetrics()
    out = m.process(sample)
    assert out.quality_metrics.lmd == pytest.approx(0.0)
    assert out.quality_metrics.f_lmd == pytest.approx(0.0)


def test_lmd_shifted_meshes_positive(tmp_path, monkeypatch):
    import ayase.modules.lmd as lmod

    m = lmod.LMDModule()
    m._backend = "mediapipe"
    m._mesh = object()
    m._lips = {0, 1, 2}
    rng = np.random.RandomState(0)
    a = rng.rand(10, 468, 3).astype(np.float32)
    b = np.roll(a, 5, axis=1)  # permute landmarks -> nonzero distance
    seqs = {"v.mp4": (a, np.arange(10)), "r.mp4": (b, np.arange(10))}
    monkeypatch.setattr("ayase.modules._mp_seq.face_mesh_seq",
                        lambda p, **kw: seqs[p.name])
    ref = tmp_path / "r.mp4"
    ref.write_bytes(b"x")
    sample = Sample(path=tmp_path / "v.mp4", is_video=True, reference_path=ref)
    sample.quality_metrics = QualityMetrics()
    out = m.process(sample)
    assert out.quality_metrics.lmd > 0
    assert out.quality_metrics.f_lmd > 0
