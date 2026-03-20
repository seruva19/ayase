"""Tests for vlm_judge module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_vlm_judge_basics():
    from ayase.modules.vlm_judge import VLMJudgeModule
    _test_module_basics(VLMJudgeModule, "vlm_judge")

def test_vlm_judge_image(image_sample):
    from ayase.modules.vlm_judge import VLMJudgeModule
    image_sample.quality_metrics = QualityMetrics()
    m = VLMJudgeModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_vlm_judge_video(video_sample):
    from ayase.modules.vlm_judge import VLMJudgeModule
    video_sample.quality_metrics = QualityMetrics()
    m = VLMJudgeModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample
