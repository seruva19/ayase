"""Tests for vqa_score module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_vqa_score_basics():
    from ayase.modules.vqa_score import VQAScoreModule
    _test_module_basics(VQAScoreModule, "vqa_score")

def test_vqa_score_image(image_sample):
    from ayase.modules.vqa_score import VQAScoreModule
    image_sample.quality_metrics = QualityMetrics()
    m = VQAScoreModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_vqa_score_video(video_sample):
    from ayase.modules.vqa_score import VQAScoreModule
    video_sample.quality_metrics = QualityMetrics()
    m = VQAScoreModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample
