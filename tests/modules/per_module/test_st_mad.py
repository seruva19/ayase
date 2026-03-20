"""Tests for st_mad module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_st_mad_basics():
    from ayase.modules.st_mad import STMADModule
    _test_module_basics(STMADModule, "st_mad")

def test_st_mad_no_reference(image_sample):
    from ayase.modules.st_mad import STMADModule
    m = STMADModule()
    result = m.process(image_sample)
    assert result is image_sample
