"""Tests for deepvqa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_deepvqa_basics():
    from ayase.modules.deepvqa import DeepVQAModule
    _test_module_basics(DeepVQAModule, "deepvqa")

def test_deepvqa_no_reference(image_sample):
    from ayase.modules.deepvqa import DeepVQAModule
    m = DeepVQAModule()
    result = m.process(image_sample)
    assert result is image_sample
