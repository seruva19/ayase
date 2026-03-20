"""Tests for psnr_div module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_psnr_div_basics():
    from ayase.modules.psnr_div import PSNRDIVModule
    _test_module_basics(PSNRDIVModule, "psnr_div")

def test_psnr_div_no_reference(image_sample):
    from ayase.modules.psnr_div import PSNRDIVModule
    m = PSNRDIVModule()
    result = m.process(image_sample)
    assert result is image_sample
