"""Tests for paranoid_decoder module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_paranoid_decoder_basics():
    from ayase.modules.paranoid_decoder import ParanoidDecoderModule
    _test_module_basics(ParanoidDecoderModule, "paranoid_decoder")

def test_paranoid_decoder_video(video_sample):
    from ayase.modules.paranoid_decoder import ParanoidDecoderModule
    video_sample.quality_metrics = QualityMetrics()
    m = ParanoidDecoderModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample
