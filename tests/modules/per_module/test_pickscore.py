"""Tests for PickScore module."""

from ayase.models import QualityMetrics

from ..conftest import _test_module_basics


def test_pickscore_basics():
    from ayase.modules.pickscore import PickScoreModule

    _test_module_basics(PickScoreModule, "pickscore")


def test_pickscore_skip_no_caption(image_sample):
    from ayase.modules.pickscore import PickScoreModule

    result = PickScoreModule().process(image_sample)
    assert result is image_sample
    if result.quality_metrics:
        assert result.quality_metrics.pickscore_score is None


def test_pickscore_skip_no_ml(image_sample):
    from ayase.modules.pickscore import PickScoreModule

    image_sample.quality_metrics = QualityMetrics()
    result = PickScoreModule().process(image_sample)
    assert result is image_sample
    assert result.quality_metrics.pickscore_score is None


def test_pickscore_field_exists():
    qm = QualityMetrics()
    assert hasattr(qm, "pickscore_score")
    assert qm.pickscore_score is None
    qm.pickscore_score = 1.5
    assert qm.pickscore_score == 1.5


def test_pickscore_field_group():
    qm = QualityMetrics()
    assert qm._FIELD_GROUPS.get("pickscore_score") == "alignment"
