"""Tests for butteraugli module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_butteraugli_basics():
    from ayase.modules.butteraugli import ButteraugliModule
    _test_module_basics(ButteraugliModule, "butteraugli")

def test_butteraugli_no_reference(image_sample):
    from ayase.modules.butteraugli import ButteraugliModule
    m = ButteraugliModule()
    result = m.process(image_sample)
    assert result is image_sample
