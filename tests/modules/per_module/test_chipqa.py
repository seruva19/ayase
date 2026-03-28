"""Tests for ChipQA module."""

from ayase.models import QualityMetrics

from ..conftest import _test_module_basics


def test_chipqa_basics():
    from ayase.modules.chipqa import ChipQAModule

    _test_module_basics(ChipQAModule, "chipqa")


def test_chipqa_skip_without_backend(video_sample):
    from ayase.modules.chipqa import ChipQAModule

    video_sample.quality_metrics = QualityMetrics()
    result = ChipQAModule().process(video_sample)
    assert result is video_sample
    assert result.quality_metrics.chipqa_score is None


def test_chipqa_field_exists():
    qm = QualityMetrics()
    assert hasattr(qm, "chipqa_score")
    assert qm.chipqa_score is None


def test_chipqa_field_group():
    qm = QualityMetrics()
    assert qm._FIELD_GROUPS.get("chipqa_score") == "nr_quality"
