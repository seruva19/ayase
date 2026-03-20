"""Tests for spherical_psnr module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_spherical_psnr_basics():
    from ayase.modules.spherical_psnr import SphericalPSNRModule
    _test_module_basics(SphericalPSNRModule, "spherical_psnr")

def test_spherical_psnr_no_reference(image_sample):
    from ayase.modules.spherical_psnr import SphericalPSNRModule
    m = SphericalPSNRModule()
    result = m.process(image_sample)
    assert result is image_sample
