"""Tests for vqa2 module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_vqa2_basics():
    from ayase.modules.vqa2 import VQA2Module
    _test_module_basics(VQA2Module, "vqa2")


def test_vqa2_provisional():
    # Real backend requires the repo's custom `llava` package (see REVIVAL
    # NOTES). Until verified end-to-end it must stay provisional.
    from ayase.modules.vqa2 import VQA2Module
    assert VQA2Module.provisional is True


def test_vqa2_wa5_range():
    # wa5 maps any five logits into the published [0.2, 1.0] score range,
    # and matches the reference weighted-softmax formula.
    from ayase.modules.vqa2 import _wa5
    assert abs(_wa5([0.0, 0.0, 0.0, 0.0, 0.0]) - 0.6) < 1e-9  # uniform -> mean weight
    for logits in ([10, 0, 0, 0, 0], [0, 0, 0, 0, 10], [1.0, -2.0, 3.0, 0.5, -1.0]):
        s = _wa5(logits)
        assert 0.2 <= s <= 1.0


def test_vqa2_image(image_sample):
    # No real `llava` backend in a standard install -> graceful unavailable,
    # metric left as None (real-or-none, no proxy).
    from ayase.modules.vqa2 import VQA2Module
    image_sample.quality_metrics = QualityMetrics()
    m = VQA2Module()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample
    if not m._ml_available:
        assert result.quality_metrics.vqa2_score is None


def test_vqa2_video(video_sample):
    from ayase.modules.vqa2 import VQA2Module
    video_sample.quality_metrics = QualityMetrics()
    m = VQA2Module()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample
    if not m._ml_available:
        assert result.quality_metrics.vqa2_score is None
