"""Tests for psnr_hvs_approx module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_psnr_hvs_basics():
    from ayase.modules.psnr_hvs_approx import PSNRHVSModule
    _test_module_basics(PSNRHVSModule, "psnr_hvs_approx")

def test_psnr_hvs_no_reference(image_sample):
    from ayase.modules.psnr_hvs_approx import PSNRHVSModule
    m = PSNRHVSModule()
    result = m.process(image_sample)
    assert result is image_sample
