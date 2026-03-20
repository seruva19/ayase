"""Tests for xpsnr module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_xpsnr_basics():
    from ayase.modules.xpsnr import XPSNRModule
    _test_module_basics(XPSNRModule, "xpsnr")

def test_xpsnr_no_reference(image_sample):
    from ayase.modules.xpsnr import XPSNRModule
    m = XPSNRModule()
    result = m.process(image_sample)
    assert result is image_sample
