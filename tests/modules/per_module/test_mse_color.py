"""Tests for pcqm module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_pcqm_basics():
    from ayase.modules.mse_color import PCQMModule
    _test_module_basics(PCQMModule, "mse_color")

def test_pcqm_no_reference(image_sample):
    from ayase.modules.mse_color import PCQMModule
    m = PCQMModule()
    result = m.process(image_sample)
    assert result is image_sample
