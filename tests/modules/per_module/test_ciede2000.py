"""Tests for ciede2000 module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_ciede2000_basics():
    from ayase.modules.ciede2000 import CIEDE2000Module
    _test_module_basics(CIEDE2000Module, "ciede2000")

def test_ciede2000_no_reference(image_sample):
    from ayase.modules.ciede2000 import CIEDE2000Module
    m = CIEDE2000Module()
    result = m.process(image_sample)
    assert result is image_sample
