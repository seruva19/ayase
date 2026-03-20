"""Tests for nemo_curator module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_nemo_curator_basics():
    from ayase.modules.nemo_curator import NemoCuratorModule
    _test_module_basics(NemoCuratorModule, "nemo_curator")

def test_nemo_curator_image(image_sample):
    from ayase.modules.nemo_curator import NemoCuratorModule
    image_sample.quality_metrics = QualityMetrics()
    m = NemoCuratorModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_nemo_curator_video(video_sample):
    from ayase.modules.nemo_curator import NemoCuratorModule
    video_sample.quality_metrics = QualityMetrics()
    m = NemoCuratorModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample
