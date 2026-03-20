"""Tests for rankdvqa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_rankdvqa_basics():
    from ayase.modules.rankdvqa import RankDVQAModule
    _test_module_basics(RankDVQAModule, "rankdvqa")

def test_rankdvqa_no_reference(image_sample):
    from ayase.modules.rankdvqa import RankDVQAModule
    m = RankDVQAModule()
    result = m.process(image_sample)
    assert result is image_sample
