"""Tests for erqa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_erqa_basics():
    from ayase.modules.erqa import ERQAModule
    _test_module_basics(ERQAModule, "erqa")

def test_erqa_no_reference(image_sample):
    from ayase.modules.erqa import ERQAModule
    m = ERQAModule()
    result = m.process(image_sample)
    assert result is image_sample
