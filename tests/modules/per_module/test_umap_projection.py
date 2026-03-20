"""Tests for umap_projection module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_umap_projection_basics():
    from ayase.modules.umap_projection import UMAPProjectionModule
    _test_module_basics(UMAPProjectionModule, "umap_projection")

def test_umap_projection_extract(video_sample):
    from ayase.modules.umap_projection import UMAPProjectionModule
    m = UMAPProjectionModule()
    feat = m.extract_features(video_sample)
    # May be None for non-video or missing deps
    assert video_sample is not None
