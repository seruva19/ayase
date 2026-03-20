"""Tests for ssimulacra2 module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_ssimulacra2_basics():
    from ayase.modules.ssimulacra2 import SSIMULACRA2Module
    _test_module_basics(SSIMULACRA2Module, "ssimulacra2")

def test_ssimulacra2_no_reference(image_sample):
    from ayase.modules.ssimulacra2 import SSIMULACRA2Module
    m = SSIMULACRA2Module()
    result = m.process(image_sample)
    assert result is image_sample
