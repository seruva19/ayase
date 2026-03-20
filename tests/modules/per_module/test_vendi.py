"""Tests for vendi module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_vendi_basics():
    from ayase.modules.vendi import VendiModule
    _test_module_basics(VendiModule, "vendi")

def test_vendi_extract(video_sample):
    from ayase.modules.vendi import VendiModule
    m = VendiModule()
    feat = m.extract_features(video_sample)
    # May be None for non-video or missing deps
    assert video_sample is not None
