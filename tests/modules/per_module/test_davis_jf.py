"""Tests for davis_jf module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_davis_jf_basics():
    from ayase.modules.davis_jf import DAVISJFModule
    _test_module_basics(DAVISJFModule, "davis_jf")

def test_davis_jf_no_reference(image_sample):
    from ayase.modules.davis_jf import DAVISJFModule
    m = DAVISJFModule()
    result = m.process(image_sample)
    assert result is image_sample
