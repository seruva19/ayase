"""Tests for vmaf_phone module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_vmaf_phone_basics():
    from ayase.modules.vmaf_phone import VMAFPhoneModule
    _test_module_basics(VMAFPhoneModule, "vmaf_phone")

def test_vmaf_phone_no_reference(image_sample):
    from ayase.modules.vmaf_phone import VMAFPhoneModule
    m = VMAFPhoneModule()
    result = m.process(image_sample)
    assert result is image_sample
