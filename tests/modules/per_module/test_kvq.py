"""Tests for kvq module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_kvq_basics():
    from ayase.modules.kvq import KVQModule
    _test_module_basics(KVQModule, "kvq")

def test_kvq_image(image_sample):
    from ayase.modules.kvq import KVQModule
    image_sample.quality_metrics = QualityMetrics()
    m = KVQModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_kvq_video(video_sample):
    from ayase.modules.kvq import KVQModule
    video_sample.quality_metrics = QualityMetrics()
    m = KVQModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample
