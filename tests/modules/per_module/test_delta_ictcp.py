"""Tests for delta_ictcp module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_delta_ictcp_basics():
    from ayase.modules.delta_ictcp import DeltaICtCpModule
    _test_module_basics(DeltaICtCpModule, "delta_ictcp")

def test_delta_ictcp_no_reference(image_sample):
    from ayase.modules.delta_ictcp import DeltaICtCpModule
    m = DeltaICtCpModule()
    result = m.process(image_sample)
    assert result is image_sample
