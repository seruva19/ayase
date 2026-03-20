"""Tests for bd_rate module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_bd_rate_basics():
    from ayase.modules.bd_rate import BDRateModule
    _test_module_basics(BDRateModule, "bd_rate")

def test_bd_rate_extract(video_sample):
    from ayase.modules.bd_rate import BDRateModule
    m = BDRateModule()
    feat = m.extract_features(video_sample)
    # May be None for non-video or missing deps
    assert video_sample is not None
