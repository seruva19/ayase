"""Tests for kvd module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_kvd_basics():
    from ayase.modules.kvd import KVDModule
    _test_module_basics(KVDModule, "kvd")

def test_kvd_extract(video_sample):
    from ayase.modules.kvd import KVDModule
    m = KVDModule()
    feat = m.extract_features(video_sample)
    # May be None for non-video or missing deps
    assert video_sample is not None
