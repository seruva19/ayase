"""Tests for vmaf_neg module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_vmaf_neg_basics():
    from ayase.modules.vmaf_neg import VMAFNEGModule
    _test_module_basics(VMAFNEGModule, "vmaf_neg")

def test_vmaf_neg_no_reference(image_sample):
    from ayase.modules.vmaf_neg import VMAFNEGModule
    m = VMAFNEGModule()
    result = m.process(image_sample)
    assert result is image_sample
