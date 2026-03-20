"""Tests for flip module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_flip_basics():
    from ayase.modules.flip_metric import FLIPModule
    _test_module_basics(FLIPModule, "flip")

def test_flip_no_reference(image_sample):
    from ayase.modules.flip_metric import FLIPModule
    m = FLIPModule()
    result = m.process(image_sample)
    assert result is image_sample
