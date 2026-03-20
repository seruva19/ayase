"""Tests for vif module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_vif_basics():
    from ayase.modules.vif import VIFModule
    _test_module_basics(VIFModule, "vif")

def test_vif_no_reference(image_sample):
    from ayase.modules.vif import VIFModule
    m = VIFModule()
    result = m.process(image_sample)
    assert result is image_sample
