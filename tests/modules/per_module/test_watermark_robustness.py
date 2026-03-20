"""Tests for watermark_robustness module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_watermark_robustness_basics():
    from ayase.modules.watermark_robustness import WatermarkRobustnessModule
    _test_module_basics(WatermarkRobustnessModule, "watermark_robustness")

def test_watermark_robustness_image(image_sample):
    from ayase.modules.watermark_robustness import WatermarkRobustnessModule
    image_sample.quality_metrics = QualityMetrics()
    m = WatermarkRobustnessModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_watermark_robustness_video(video_sample):
    from ayase.modules.watermark_robustness import WatermarkRobustnessModule
    video_sample.quality_metrics = QualityMetrics()
    m = WatermarkRobustnessModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample
