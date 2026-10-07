"""Tests for the aed_apd module (PIRenderer AED/APD on TDDFA coefficients)."""

import numpy as np
import pytest

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics, Sample


def test_aed_apd_basics():
    from ayase.modules.aed_apd import AedApdModule
    _test_module_basics(AedApdModule, "aed_apd")


def test_aed_apd_provenance_adapted():
    from ayase.modules.aed_apd import AedApdModule
    meta = AedApdModule.get_metadata()
    assert meta["provenance"]["aed"] == "adapted"
    assert meta["provenance"]["apd"] == "adapted"
    assert "aed" in meta["deviations"] and "apd" in meta["deviations"]
    assert "PIRenderer" in meta["sources"]["aed"]


def test_aed_apd_without_backend_leaves_fields_unset(tmp_path):
    from ayase.modules.aed_apd import AedApdModule
    m = AedApdModule()
    sample = Sample(path=tmp_path / "v.mp4", is_video=True)
    sample.quality_metrics = QualityMetrics()
    assert m.process(sample).quality_metrics.aed is None
    assert m.process(sample).quality_metrics.apd is None


def test_aed_apd_no_reference_skips(tmp_path):
    from ayase.modules.aed_apd import AedApdModule
    m = AedApdModule()
    m._backend = "tddfa"
    m._extractor = object()
    sample = Sample(path=tmp_path / "v.mp4", is_video=True)
    sample.quality_metrics = QualityMetrics()
    assert m.process(sample).quality_metrics.aed is None


def test_aed_apd_self_reference_still_valid(tmp_path):
    """reference_path may coincide with path; extraction is mocked anyway."""
    from ayase.modules.aed_apd import AedApdModule
    m = AedApdModule()
    m._backend = "tddfa"
    rng = np.random.RandomState(0)
    cand = rng.rand(10, 62).astype(np.float32)
    ref = cand + 0.5
    inds = np.arange(10)
    m._extractor = type("E", (), {"extract": staticmethod(
        lambda p: (cand, inds) if p.name == "v.mp4" else (ref, inds))})()
    ref_file = tmp_path / "r.mp4"
    ref_file.write_bytes(b"x")
    sample = Sample(path=tmp_path / "v.mp4", is_video=True,
                    reference_path=ref_file)
    sample.quality_metrics = QualityMetrics()
    out = m.process(sample)
    assert out.quality_metrics.aed == pytest.approx(0.5)
    assert out.quality_metrics.apd == pytest.approx(0.5)


def test_aed_apd_alignment_truncates_to_shorter(tmp_path):
    from ayase.modules.aed_apd import AedApdModule
    m = AedApdModule()
    m._backend = "tddfa"
    cand = np.zeros((4, 62), dtype=np.float32)
    ref = np.ones((9, 62), dtype=np.float32)
    inds = np.arange(9)
    m._extractor = type("E", (), {"extract": staticmethod(
        lambda p: (cand, np.arange(4)) if p.name == "v.mp4" else (ref, inds))})()
    ref_file = tmp_path / "r.mp4"
    ref_file.write_bytes(b"x")
    sample = Sample(path=tmp_path / "v.mp4", is_video=True,
                    reference_path=ref_file)
    sample.quality_metrics = QualityMetrics()
    out = m.process(sample)
    assert out.quality_metrics.aed == pytest.approx(1.0)
    assert out.quality_metrics.apd == pytest.approx(1.0)


def test_aed_apd_image_input_skips(tmp_path):
    from ayase.modules.aed_apd import AedApdModule
    m = AedApdModule()
    m._backend = "tddfa"
    m._extractor = object()
    sample = Sample(path=tmp_path / "i.png", is_video=False)
    assert m.process(sample).quality_metrics is None
