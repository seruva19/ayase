"""Tests for hdr_vdp module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_hdr_vdp_basics():
    from ayase.modules.hdr_vdp import HDRVDPModule
    _test_module_basics(HDRVDPModule, "hdr_vdp")

def test_hdr_vdp_no_reference(image_sample):
    from ayase.modules.hdr_vdp import HDRVDPModule
    m = HDRVDPModule()
    result = m.process(image_sample)
    assert result is image_sample
