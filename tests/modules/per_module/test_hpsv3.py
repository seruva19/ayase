"""Tests for HPSv3 module."""

from ayase.models import QualityMetrics

from ..conftest import _test_module_basics


def test_hpsv3_basics():
    from ayase.modules.hpsv3 import HPSv3Module

    _test_module_basics(HPSv3Module, "hpsv3")


def test_hpsv3_skip_no_caption(image_sample):
    from ayase.modules.hpsv3 import HPSv3Module

    result = HPSv3Module().process(image_sample)
    assert result is image_sample
    if result.quality_metrics:
        assert result.quality_metrics.hpsv3_score is None


def test_hpsv3_skip_no_ml(image_sample):
    from ayase.modules.hpsv3 import HPSv3Module

    image_sample.quality_metrics = QualityMetrics()
    result = HPSv3Module().process(image_sample)
    assert result is image_sample
    assert result.quality_metrics.hpsv3_score is None


def test_hpsv3_field_exists():
    qm = QualityMetrics()
    assert hasattr(qm, "hpsv3_score")
    assert qm.hpsv3_score is None
    qm.hpsv3_score = 1.5
    assert qm.hpsv3_score == 1.5


def test_hpsv3_field_group():
    qm = QualityMetrics()
    assert qm._FIELD_GROUPS.get("hpsv3_score") == "alignment"
