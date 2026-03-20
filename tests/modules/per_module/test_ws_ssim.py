"""Tests for ws_ssim module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_ws_ssim_basics():
    from ayase.modules.ws_ssim import WSSSIMModule
    _test_module_basics(WSSSIMModule, "ws_ssim")

def test_ws_ssim_no_reference(image_sample):
    from ayase.modules.ws_ssim import WSSSIMModule
    m = WSSSIMModule()
    result = m.process(image_sample)
    assert result is image_sample
