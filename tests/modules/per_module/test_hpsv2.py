"""Tests for HPSv2 module."""

from ..conftest import _test_module_basics


def test_hpsv2_basics():
    from ayase.modules.hpsv2 import HPSv2Module

    _test_module_basics(HPSv2Module, "hpsv2")


def test_hpsv2_skip_no_backend(image_sample):
    from ayase.models import CaptionMetadata
    from ayase.modules.hpsv2 import HPSv2Module

    image_sample.caption = CaptionMetadata(text="a red apple", length=11)
    result = HPSv2Module().process(image_sample)

    assert result is image_sample
    if result.quality_metrics is not None:
        assert result.quality_metrics.hpsv2_score is None


def test_hpsv2_field_exists_and_grouped():
    from ayase.models import QualityMetrics

    qm = QualityMetrics()
    assert qm.hpsv2_score is None
    qm.hpsv2_score = 30.5
    assert qm.hpsv2_score == 30.5
    assert qm._FIELD_GROUPS.get("hpsv2_score") == "alignment"


def test_hpsv2_as_floats():
    from ayase.modules.hpsv2 import _as_floats

    assert _as_floats([1, "2.5", None]) == [1.0, 2.5]

