"""Tests for watermark_robustness module."""

import numpy as np

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


def test_retention_score():
    from ayase.modules.watermark_robustness import WatermarkRobustnessModule

    score = WatermarkRobustnessModule._retention_score(0.8, [0.8, 0.4, 1.0])
    assert score == (1.0 + 0.5 + 1.0) / 3.0


def test_attack_suite_is_deterministic():
    from ayase.modules.watermark_robustness import WatermarkRobustnessModule

    rng = np.random.default_rng(12)
    frames = [rng.integers(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(4)]
    module = WatermarkRobustnessModule()
    clean = [module._score_frame(frame) for frame in frames]
    first = module._robustness_score(frames, clean, include_temporal=True)
    second = module._robustness_score(frames, clean, include_temporal=True)
    assert first == second
    assert 0.0 <= first <= 1.0
