"""Tests for vqinsight module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_vqinsight_basics():
    from ayase.modules.vqinsight import VQInsightModule
    _test_module_basics(VQInsightModule, "vqinsight")

def test_vqinsight_image(image_sample):
    from ayase.modules.vqinsight import VQInsightModule
    image_sample.quality_metrics = QualityMetrics()
    m = VQInsightModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_vqinsight_video(video_sample):
    from ayase.modules.vqinsight import VQInsightModule
    video_sample.quality_metrics = QualityMetrics()
    m = VQInsightModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample
