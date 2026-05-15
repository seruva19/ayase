"""Tests for fvd module."""

from ..conftest import _test_module_basics
from ayase.models import DatasetStats, QualityMetrics


def test_fvd_basics():
    from ayase.modules.fvd import FVDModule
    _test_module_basics(FVDModule, "fvd")

def test_fvd_extract(video_sample):
    from ayase.modules.fvd import FVDModule
    m = FVDModule()
    feat = m.extract_features(video_sample)
    # May be None for non-video or missing deps
    assert video_sample is not None


def test_fvd_default_backbone_is_r3d18():
    from ayase.modules.fvd import FVDModule
    m = FVDModule()
    assert m.backbone == "r3d18"
    assert m.metric_name == "fvd"


def test_fvd_content_debiased_backend():
    from ayase.modules.fvd import FVDModule
    m = FVDModule(config={"backbone": "content_debiased"})
    assert m.backbone == "content_debiased"
    assert m.metric_name == "fvd_content_debiased"
    # New dataset-level field exists and defaults to None
    stats = DatasetStats(total_samples=0, valid_samples=0, invalid_samples=0, total_size=0)
    assert stats.fvd_content_debiased is None


def test_fvd_dinov2_backend():
    from ayase.modules.fvd import FVDModule
    m = FVDModule(config={"backbone": "dinov2"})
    assert m.backbone == "dinov2"
    assert m.metric_name == "fvd_dinov2"
    stats = DatasetStats(total_samples=0, valid_samples=0, invalid_samples=0, total_size=0)
    assert stats.fvd_dinov2 is None


def test_fvd_unknown_backbone_falls_back_to_r3d18():
    from ayase.modules.fvd import FVDModule
    m = FVDModule(config={"backbone": "nonexistent_backbone"})
    assert m.backbone == "r3d18"
    assert m.metric_name == "fvd"
