"""Tests for aucon_prmse (MarioNETte AUCON/PRMSE via Py-Feat)."""

import numpy as np
import pytest

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics, Sample


def test_aucon_prmse_basics():
    from ayase.modules.aucon_prmse import AuconPrmseModule
    _test_module_basics(AuconPrmseModule, "aucon_prmse")


def test_aucon_prmse_provenance():
    from ayase.modules.aucon_prmse import AuconPrmseModule
    meta = AuconPrmseModule.get_metadata()
    assert meta["provenance"]["aucon"] == "adapted"
    assert meta["provenance"]["prmse"] == "adapted"
    assert "MarioNETte" in meta["sources"]["aucon"]
    assert "Py-Feat" in meta["deviations"]["aucon"]


def test_aucon_prmse_no_backend(tmp_path):
    from ayase.modules.aucon_prmse import AuconPrmseModule
    m = AuconPrmseModule()
    sample = Sample(path=tmp_path / "v.mp4", is_video=True)
    sample.quality_metrics = QualityMetrics()
    assert m.process(sample).quality_metrics.aucon is None
    assert m.process(sample).quality_metrics.prmse is None


def test_aucon_prmse_no_reference_skips(tmp_path):
    from ayase.modules.aucon_prmse import AuconPrmseModule
    m = AuconPrmseModule()
    m._backend = "pyfeat"
    m._detector = object()
    sample = Sample(path=tmp_path / "v.mp4", is_video=True)
    sample.quality_metrics = QualityMetrics()
    assert m.process(sample).quality_metrics.aucon is None


def test_aucon_prmse_identical_streams(tmp_path):
    from ayase.modules.aucon_prmse import AuconPrmseModule
    m = AuconPrmseModule()
    m._backend = "pyfeat"
    m._detector = object()
    aus = np.array([[True, False, True]] * 5)
    poses = np.zeros((5, 3), dtype=np.float32)
    m._detect = lambda p: (aus, poses)
    ref_file = tmp_path / "r.mp4"
    ref_file.write_bytes(b"x")
    sample = Sample(path=tmp_path / "v.mp4", is_video=True,
                    reference_path=ref_file)
    sample.quality_metrics = QualityMetrics()
    out = m.process(sample)
    assert out.quality_metrics.aucon == pytest.approx(1.0)
    assert out.quality_metrics.prmse == pytest.approx(0.0)


def test_aucon_prmse_mismatch(tmp_path):
    from ayase.modules.aucon_prmse import AuconPrmseModule
    m = AuconPrmseModule()
    m._backend = "pyfeat"
    m._detector = object()
    aus_a = np.array([[True, False]] * 4)
    aus_b = np.array([[True, False]] * 2 + [[False, True]] * 2)
    poses_a = np.zeros((4, 3), dtype=np.float32)
    poses_b = np.ones((4, 3), dtype=np.float32) * 3.0
    seqs = {"v.mp4": (aus_a, poses_a), "r.mp4": (aus_b, poses_b)}
    m._detect = lambda p: seqs[p.name]
    ref_file = tmp_path / "r.mp4"
    ref_file.write_bytes(b"x")
    sample = Sample(path=tmp_path / "v.mp4", is_video=True,
                    reference_path=ref_file)
    sample.quality_metrics = QualityMetrics()
    out = m.process(sample)
    assert out.quality_metrics.aucon == pytest.approx(0.5)
    assert out.quality_metrics.prmse == pytest.approx(3.0)
