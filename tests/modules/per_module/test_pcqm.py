"""Tests for pcqm module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_pcqm_basics():
    from ayase.modules.pcqm import PCQMModule
    _test_module_basics(PCQMModule, "pcqm")

def test_pcqm_no_reference(image_sample):
    from ayase.modules.pcqm import PCQMModule
    m = PCQMModule()
    result = m.process(image_sample)
    assert result is image_sample
