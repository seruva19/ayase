"""Tests for pvmaf module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_pvmaf_basics():
    from ayase.modules.pvmaf import PVMAFModule
    _test_module_basics(PVMAFModule, "pvmaf")

def test_pvmaf_no_reference(image_sample):
    from ayase.modules.pvmaf import PVMAFModule
    m = PVMAFModule()
    result = m.process(image_sample)
    assert result is image_sample
