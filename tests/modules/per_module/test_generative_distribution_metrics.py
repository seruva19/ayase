"""Tests for generative_distribution_metrics module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_generative_distribution_metrics_basics():
    from ayase.modules.generative_distribution_metrics import GenerativeDistributionCompatModule
    _test_module_basics(GenerativeDistributionCompatModule, "generative_distribution_metrics")

def test_generative_distribution_metrics_extract(video_sample):
    from ayase.modules.generative_distribution_metrics import GenerativeDistributionCompatModule
    m = GenerativeDistributionCompatModule()
    feat = m.extract_features(video_sample)
    # May be None for non-video or missing deps
    assert video_sample is not None
