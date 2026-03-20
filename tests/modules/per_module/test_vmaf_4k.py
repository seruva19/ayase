"""Tests for vmaf_4k module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_vmaf_4k_basics():
    from ayase.modules.vmaf_4k import VMAF4KModule
    _test_module_basics(VMAF4KModule, "vmaf_4k")

def test_vmaf_4k_no_reference(image_sample):
    from ayase.modules.vmaf_4k import VMAF4KModule
    m = VMAF4KModule()
    result = m.process(image_sample)
    assert result is image_sample
