"""Tests for flip_metric module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_flip_metric_basics():
    from ayase.modules.flip_metric import FLIPCompatModule
    _test_module_basics(FLIPCompatModule, "flip_metric")

def test_flip_metric_no_reference(image_sample):
    from ayase.modules.flip_metric import FLIPCompatModule
    m = FLIPCompatModule()
    result = m.process(image_sample)
    assert result is image_sample
