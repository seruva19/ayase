"""Tests for strred module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_strred_basics():
    from ayase.modules.strred import STRREDModule
    _test_module_basics(STRREDModule, "strred")

def test_strred_no_reference(image_sample):
    from ayase.modules.strred import STRREDModule
    m = STRREDModule()
    result = m.process(image_sample)
    assert result is image_sample
