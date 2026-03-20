"""Tests for vqathinker module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_vqathinker_basics():
    from ayase.modules.vqathinker import VQAThinkerModule
    _test_module_basics(VQAThinkerModule, "vqathinker")

def test_vqathinker_image(image_sample):
    from ayase.modules.vqathinker import VQAThinkerModule
    image_sample.quality_metrics = QualityMetrics()
    m = VQAThinkerModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_vqathinker_video(video_sample):
    from ayase.modules.vqathinker import VQAThinkerModule
    video_sample.quality_metrics = QualityMetrics()
    m = VQAThinkerModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample
