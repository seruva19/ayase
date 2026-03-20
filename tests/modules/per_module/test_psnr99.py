"""Tests for psnr99 module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_psnr99_basics():
    from ayase.modules.psnr99 import PSNR99Module
    _test_module_basics(PSNR99Module, "psnr99")

def test_psnr99_no_reference(image_sample):
    from ayase.modules.psnr99 import PSNR99Module
    m = PSNR99Module()
    result = m.process(image_sample)
    assert result is image_sample
