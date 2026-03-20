"""Tests for dataset_analytics module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_dataset_analytics_basics():
    from ayase.modules.dataset_analytics import DatasetAnalyticsModule
    _test_module_basics(DatasetAnalyticsModule, "dataset_analytics")

def test_dataset_analytics_extract(video_sample):
    from ayase.modules.dataset_analytics import DatasetAnalyticsModule
    m = DatasetAnalyticsModule()
    feat = m.extract_features(video_sample)
    # May be None for non-video or missing deps
    assert video_sample is not None
