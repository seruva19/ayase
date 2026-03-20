"""Tests for ms_ssim module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_ms_ssim_basics():
    from ayase.modules.ms_ssim import MSSSIMModule
    _test_module_basics(MSSSIMModule, "ms_ssim")

def test_ms_ssim_no_reference(image_sample):
    from ayase.modules.ms_ssim import MSSSIMModule
    m = MSSSIMModule()
    result = m.process(image_sample)
    assert result is image_sample
