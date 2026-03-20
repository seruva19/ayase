"""Tests for decoder_stress module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_decoder_stress_basics():
    from ayase.modules.decoder_stress import DecoderStressModule
    _test_module_basics(DecoderStressModule, "decoder_stress")

def test_decoder_stress_video(video_sample):
    from ayase.modules.decoder_stress import DecoderStressModule
    video_sample.quality_metrics = QualityMetrics()
    m = DecoderStressModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample
