"""Tests for maniqa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_maniqa_basics():
    from ayase.modules.maniqa import MANIQAModule
    _test_module_basics(MANIQAModule, "maniqa")

def test_maniqa_image(image_sample):
    from ayase.modules.maniqa import MANIQAModule
    image_sample.quality_metrics = QualityMetrics()
    m = MANIQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_maniqa_video(video_sample):
    from ayase.modules.maniqa import MANIQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = MANIQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample


def test_maniqa_frame_batch_and_serial_fallback(monkeypatch, video_sample):
    import numpy as np
    import pytest

    torch = pytest.importorskip("torch")
    from ayase.modules.maniqa import MANIQAModule

    frames = [np.full((4, 5, 3), value, dtype=np.uint8) for value in (0, 255)]
    calls = []

    def predict(batch):
        calls.append(len(batch))
        return batch.mean(dim=(1, 2, 3))

    module = MANIQAModule({"batch_frames": True})
    module._ml_available = True
    monkeypatch.setattr(module, "_load_frames", lambda sample: frames)
    module._model = predict
    assert module.process(video_sample).quality_metrics.maniqa_score == 0.5
    assert calls == [2]

    def single_only(batch):
        if len(batch) > 1:
            calls.append(len(batch))
            return torch.zeros(())
        return predict(batch)

    calls.clear()
    module._model = single_only
    assert module.process(video_sample).quality_metrics.maniqa_score == 0.5
    assert calls == [2, 1, 1]
