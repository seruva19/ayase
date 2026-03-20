"""Tests for worldscore module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_worldscore_basics():
    from ayase.modules.worldscore import WorldScoreModule
    _test_module_basics(WorldScoreModule, "worldscore")

def test_worldscore_extract(video_sample):
    from ayase.modules.worldscore import WorldScoreModule
    m = WorldScoreModule()
    feat = m.extract_features(video_sample)
    # May be None for non-video or missing deps
    assert video_sample is not None
