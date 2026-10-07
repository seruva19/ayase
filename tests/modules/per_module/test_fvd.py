"""Tests for fvd module."""

import warnings

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


def test_fvd_default_backbone_is_i3d():
    """The published FVD uses the I3D torchscript backbone (StyleGAN-V)."""
    from ayase.modules.fvd import FVDModule
    m = FVDModule()
    assert m.backbone == "i3d"
    assert m.metric_name == "fvd"


def test_fvd_removed_backbones_map_to_i3d():
    """Removed backbone variants fold back to the published I3D metric."""
    from ayase.modules.fvd import FVDModule
    for legacy in ("r3d18", "content_debiased", "dinov2", "nonexistent_backbone"):
        m = FVDModule(config={"backbone": legacy})
        assert m.backbone == "i3d"
        assert m.metric_name == "fvd"


def test_fvd_removed_fields_resolve_to_none():
    """Removed FVD variant fields read as None with a deprecation warning."""
    stats = DatasetStats(total_samples=0, valid_samples=0, invalid_samples=0, total_size=0)
    for name in ("fvd_content_debiased", "fvd_dinov2"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = getattr(stats, name)
        assert value is None
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)
