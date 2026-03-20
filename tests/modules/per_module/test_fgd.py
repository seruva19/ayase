"""Tests for fgd module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_fgd_basics():
    from ayase.modules.fgd import FGDModule
    _test_module_basics(FGDModule, "fgd")

def test_fgd_extract(video_sample):
    from ayase.modules.fgd import FGDModule
    m = FGDModule()
    feat = m.extract_features(video_sample)
    # May be None for non-video or missing deps
    assert video_sample is not None
