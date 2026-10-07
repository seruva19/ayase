"""Tests for spherical_psnr module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_spherical_psnr_basics():
    from ayase.modules.erp_psnr import SphericalPSNRModule
    _test_module_basics(SphericalPSNRModule, "erp_psnr")

def test_spherical_psnr_no_reference(image_sample):
    from ayase.modules.erp_psnr import SphericalPSNRModule
    m = SphericalPSNRModule()
    result = m.process(image_sample)
    assert result is image_sample
