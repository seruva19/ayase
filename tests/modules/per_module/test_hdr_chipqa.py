"""Tests for HDR-ChipQA module."""

from ayase.models import QualityMetrics

from ..conftest import _test_module_basics


def test_hdr_chipqa_basics():
    from ayase.modules.hdr_chipqa import HDRChipQAModule

    _test_module_basics(HDRChipQAModule, "hdr_chipqa")


def test_hdr_chipqa_skip_without_backend(video_sample):
    from ayase.modules.hdr_chipqa import HDRChipQAModule

    video_sample.quality_metrics = QualityMetrics()
    result = HDRChipQAModule().process(video_sample)
    assert result is video_sample
    assert result.quality_metrics.hdr_chipqa_score is None


def test_hdr_chipqa_field_exists():
    qm = QualityMetrics()
    assert hasattr(qm, "hdr_chipqa_score")
    assert qm.hdr_chipqa_score is None


def test_hdr_chipqa_field_group():
    qm = QualityMetrics()
    assert qm._FIELD_GROUPS.get("hdr_chipqa_score") == "hdr"
