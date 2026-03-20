"""Tests for pc_psnr module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_pc_psnr_basics():
    from ayase.modules.pc_psnr import PCPSNRModule
    _test_module_basics(PCPSNRModule, "pc_psnr")

def test_pc_psnr_no_reference(image_sample):
    from ayase.modules.pc_psnr import PCPSNRModule
    m = PCPSNRModule()
    result = m.process(image_sample)
    assert result is image_sample
