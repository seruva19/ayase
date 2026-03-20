"""Tests for vmaf module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_vmaf_basics():
    from ayase.modules.vmaf import VMAFModule
    _test_module_basics(VMAFModule, "vmaf")

def test_vmaf_no_reference(image_sample):
    from ayase.modules.vmaf import VMAFModule
    m = VMAFModule()
    result = m.process(image_sample)
    assert result is image_sample
