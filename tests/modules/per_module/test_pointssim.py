"""Tests for pointssim module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_pointssim_basics():
    from ayase.modules.pointssim import PointSSIMModule
    _test_module_basics(PointSSIMModule, "pointssim")

def test_pointssim_no_reference(image_sample):
    from ayase.modules.pointssim import PointSSIMModule
    m = PointSSIMModule()
    result = m.process(image_sample)
    assert result is image_sample
