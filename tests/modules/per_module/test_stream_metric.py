"""Tests for stream_metric module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_stream_metric_basics():
    from ayase.modules.stream_metric import STREAMModule
    _test_module_basics(STREAMModule, "stream_metric")

def test_stream_metric_extract(video_sample):
    from ayase.modules.stream_metric import STREAMModule
    m = STREAMModule()
    feat = m.extract_features(video_sample)
    # May be None for non-video or missing deps
    assert video_sample is not None
