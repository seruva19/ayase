"""Tests for BrightRate module."""

from ayase.models import QualityMetrics

from ..conftest import _test_module_basics


def test_brightrate_basics():
    from ayase.modules.brightrate import BrightRateModule

    _test_module_basics(BrightRateModule, "brightrate")


def test_brightrate_skip_without_backend(video_sample):
    from ayase.modules.brightrate import BrightRateModule

    video_sample.quality_metrics = QualityMetrics()
    result = BrightRateModule().process(video_sample)
    assert result is video_sample
    assert result.quality_metrics.brightrate_score is None


def test_brightrate_field_exists():
    qm = QualityMetrics()
    assert hasattr(qm, "brightrate_score")
    assert qm.brightrate_score is None


def test_brightrate_field_group():
    qm = QualityMetrics()
    assert qm._FIELD_GROUPS.get("brightrate_score") == "hdr"
