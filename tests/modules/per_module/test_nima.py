"""Tests for nima module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_nima_basics():
    from ayase.modules.nima import NIMAModule
    _test_module_basics(NIMAModule, "nima")

def test_nima_image(image_sample):
    from ayase.modules.nima import NIMAModule
    image_sample.quality_metrics = QualityMetrics()
    m = NIMAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_nima_video(video_sample):
    from ayase.modules.nima import NIMAModule
    video_sample.quality_metrics = QualityMetrics()
    m = NIMAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

def test_nima_frame_batch_replays_unusable_output(monkeypatch, video_sample):
    import numpy as np
    import pytest

    torch = pytest.importorskip("torch")
    from ayase.modules.nima import NIMAModule

    module = NIMAModule({"batch_frames": True})
    module._ml_available = True
    frames = [np.full((4, 5, 3), value, dtype=np.uint8) for value in (0, 255)]
    calls = []
    monkeypatch.setattr(module, "_load_frames", lambda sample: frames)

    def predict(batch):
        calls.append(len(batch))
        return batch.mean(dim=(1, 2, 3))

    module._model = predict
    assert module.process(video_sample).quality_metrics.nima_score == 0.5
    assert calls == [2]

    def invalid_batch(batch):
        if len(batch) > 1:
            calls.append(len(batch))
            return torch.zeros(())
        return predict(batch)

    calls.clear()
    module._model = invalid_batch
    assert module.process(video_sample).quality_metrics.nima_score == 0.5
    assert calls == [2, 1, 1]
