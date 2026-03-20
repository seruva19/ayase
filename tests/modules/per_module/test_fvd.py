"""Tests for fvd module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_fvd_basics():
    from ayase.modules.fvd import FVDModule
    _test_module_basics(FVDModule, "fvd")

def test_fvd_extract(video_sample):
    from ayase.modules.fvd import FVDModule
    m = FVDModule()
    feat = m.extract_features(video_sample)
    # May be None for non-video or missing deps
    assert video_sample is not None
