"""Tests for head_pose_diversity (SadTalker Diversity on TDDFA pose coeffs)."""

import numpy as np
import pytest

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics, Sample


def test_head_pose_diversity_basics():
    from ayase.modules.head_pose_diversity import HeadPoseDiversityModule
    _test_module_basics(HeadPoseDiversityModule, "head_pose_diversity")


def test_head_pose_diversity_provenance():
    from ayase.modules.head_pose_diversity import HeadPoseDiversityModule
    meta = HeadPoseDiversityModule.get_metadata()
    assert meta["provenance"]["head_pose_diversity"] == "adapted"
    assert "SadTalker" in meta["sources"]["head_pose_diversity"]


def test_head_pose_diversity_no_backend(tmp_path):
    from ayase.modules.head_pose_diversity import HeadPoseDiversityModule
    m = HeadPoseDiversityModule()
    sample = Sample(path=tmp_path / "v.mp4", is_video=True)
    sample.quality_metrics = QualityMetrics()
    assert m.process(sample).quality_metrics.head_pose_diversity is None


def test_head_pose_diversity_std_math(tmp_path):
    from ayase.modules.head_pose_diversity import HeadPoseDiversityModule
    m = HeadPoseDiversityModule()
    m._backend = "tddfa"
    params = np.zeros((20, 62), dtype=np.float32)
    params[:, 0] = np.linspace(0.0, 1.0, 20)  # pose dim 0 ramps
    expected = float(np.std(params[:, 0:12], axis=0).mean())
    m._extractor = type("E", (), {
        "extract": staticmethod(lambda p: (params, np.arange(20)))})()
    sample = Sample(path=tmp_path / "v.mp4", is_video=True)
    sample.quality_metrics = QualityMetrics()
    out = m.process(sample)
    assert out.quality_metrics.head_pose_diversity == pytest.approx(expected)


def test_head_pose_diversity_constant_pose_zero(tmp_path):
    from ayase.modules.head_pose_diversity import HeadPoseDiversityModule
    m = HeadPoseDiversityModule()
    m._backend = "tddfa"
    params = np.ones((15, 62), dtype=np.float32)
    m._extractor = type("E", (), {
        "extract": staticmethod(lambda p: (params, np.arange(15)))})()
    sample = Sample(path=tmp_path / "v.mp4", is_video=True)
    sample.quality_metrics = QualityMetrics()
    out = m.process(sample)
    assert out.quality_metrics.head_pose_diversity == pytest.approx(0.0)


def test_head_pose_diversity_image_skips(tmp_path):
    from ayase.modules.head_pose_diversity import HeadPoseDiversityModule
    m = HeadPoseDiversityModule()
    m._backend = "tddfa"
    m._extractor = object()
    sample = Sample(path=tmp_path / "i.png", is_video=False)
    assert m.process(sample).quality_metrics is None
