"""Tests for cgvqm module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_cgvqm_basics():
    from ayase.modules.cgvqm import CGVQMModule
    _test_module_basics(CGVQMModule, "cgvqm")

def test_cgvqm_no_reference(image_sample):
    from ayase.modules.cgvqm import CGVQMModule
    m = CGVQMModule()
    result = m.process(image_sample)
    assert result is image_sample
