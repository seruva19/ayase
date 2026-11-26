"""Tests for dreamsim_metric module."""

import pytest

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_dreamsim_metric_basics():
    from ayase.modules.dreamsim_metric import DreamSimCompatModule
    _test_module_basics(DreamSimCompatModule, "dreamsim_metric")

def test_dreamsim_metric_image(image_sample):
    from ayase.modules.dreamsim_metric import DreamSimCompatModule
    image_sample.quality_metrics = QualityMetrics()
    image_sample.reference_path = image_sample.path  # exercise the image-vs-reference path
    m = DreamSimCompatModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample
    if m._ml_available:
        assert image_sample.quality_metrics.dreamsim is not None

def test_dreamsim_metric_video(video_sample):
    from ayase.modules.dreamsim_metric import DreamSimCompatModule
    video_sample.quality_metrics = QualityMetrics()
    video_sample.reference_path = video_sample.path  # video sample + video reference
    m = DreamSimCompatModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample
    if m._ml_available:
        assert video_sample.quality_metrics.dreamsim is not None


@pytest.mark.parametrize("fail_batch", [False, True])
def test_dreamsim_batches_frame_pairs_without_changing_sample_pooling(monkeypatch, fail_batch):
    import torch

    from ayase.models import Sample
    from ayase.modules.dreamsim_metric import DreamSimCompatModule

    module = DreamSimCompatModule()
    module._ml_available = True
    frame_sets = {"ref": [1], "a": [3, 5], "b": [8]}
    monkeypatch.setattr(module, "_load_frames", lambda path: frame_sets.get(path.stem, []))
    monkeypatch.setattr(module, "_prep_batch", lambda frames: torch.tensor(frames).float())
    calls = []

    def model(first, second):
        assert torch.is_inference_mode_enabled()
        calls.append(len(first))
        if fail_batch and len(first) > 2:
            raise RuntimeError("batch unavailable")
        return (first - second).abs()

    module._model = model
    samples = [
        Sample(path="a.mp4", reference_path="ref.png", is_video=True),
        Sample(path="b.png", reference_path="ref.png", is_video=False),
        Sample(path="no-reference.png", is_video=False),
    ]
    expected = [module.process(sample.model_copy(deep=True)) for sample in samples]
    calls.clear()

    assert module.process_batch(samples) == expected
    assert calls == ([3, 2, 1] if fail_batch else [3])
