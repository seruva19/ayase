"""Tests for vfips module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_vfips_basics():
    from ayase.modules.vfips import VFIPSModule
    _test_module_basics(VFIPSModule, "vfips")

def test_vfips_no_reference(image_sample):
    from ayase.modules.vfips import VFIPSModule
    m = VFIPSModule()
    result = m.process(image_sample)
    assert result is image_sample
