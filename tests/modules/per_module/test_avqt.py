"""Tests for avqt module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_avqt_basics():
    from ayase.modules.avqt import AVQTModule
    _test_module_basics(AVQTModule, "avqt")

def test_avqt_no_reference(image_sample):
    from ayase.modules.avqt import AVQTModule
    m = AVQTModule()
    result = m.process(image_sample)
    assert result is image_sample
