"""Tests for dreamsim module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_dreamsim_basics():
    from ayase.modules.dreamsim_metric import DreamSimModule
    _test_module_basics(DreamSimModule, "dreamsim")

def test_dreamsim_image(image_sample):
    from ayase.modules.dreamsim_metric import DreamSimModule
    image_sample.quality_metrics = QualityMetrics()
    image_sample.reference_path = image_sample.path  # exercise the image-vs-reference path
    m = DreamSimModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample
    # When the backend loads, the metric must actually be computed. A bare
    # `result is sample` assertion silently passed even while the forward raised
    # (preprocess batching / device mismatch) and the metric stayed None.
    if m._ml_available:
        assert image_sample.quality_metrics.dreamsim is not None

def test_dreamsim_video(video_sample):
    from ayase.modules.dreamsim_metric import DreamSimModule
    video_sample.quality_metrics = QualityMetrics()
    video_sample.reference_path = video_sample.path  # video sample + video reference
    m = DreamSimModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample
    if m._ml_available:
        assert video_sample.quality_metrics.dreamsim is not None
