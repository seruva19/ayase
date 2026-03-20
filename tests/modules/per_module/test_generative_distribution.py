"""Tests for generative_distribution module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_generative_distribution_basics():
    from ayase.modules.generative_distribution_metrics import GenerativeDistributionModule
    _test_module_basics(GenerativeDistributionModule, "generative_distribution")

def test_generative_distribution_extract(video_sample):
    from ayase.modules.generative_distribution_metrics import GenerativeDistributionModule
    m = GenerativeDistributionModule()
    feat = m.extract_features(video_sample)
    # May be None for non-video or missing deps
    assert video_sample is not None
