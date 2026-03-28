"""Tests for HDRMAX module."""

from ayase.models import QualityMetrics

from ..conftest import _test_module_basics


def test_hdrmax_basics():
    from ayase.modules.hdrmax import HDRMAXModule

    _test_module_basics(HDRMAXModule, "hdrmax")


def test_hdrmax_skip_without_reference(video_sample):
    from ayase.modules.hdrmax import HDRMAXModule

    video_sample.quality_metrics = QualityMetrics()
    result = HDRMAXModule().process(video_sample)
    assert result is video_sample
    assert result.quality_metrics.hdrmax_score is None


def test_hdrmax_field_exists():
    qm = QualityMetrics()
    assert hasattr(qm, "hdrmax_score")
    assert qm.hdrmax_score is None


def test_hdrmax_field_group():
    qm = QualityMetrics()
    assert qm._FIELD_GROUPS.get("hdrmax_score") == "hdr"
