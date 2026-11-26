"""Tests for clip_iqa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_clip_iqa_basics():
    from ayase.modules.clip_iqa import CLIPIQAModule
    _test_module_basics(CLIPIQAModule, "clip_iqa")

def test_clip_iqa_image(image_sample):
    from ayase.modules.clip_iqa import CLIPIQAModule
    image_sample.quality_metrics = QualityMetrics()
    m = CLIPIQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_clip_iqa_video(video_sample):
    from ayase.modules.clip_iqa import CLIPIQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = CLIPIQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample


def test_clip_iqa_batches_matching_shapes_and_preserves_warning_threshold(monkeypatch):
    import numpy as np
    import torch

    from ayase.models import Sample
    from ayase.modules.clip_iqa import CLIPIQAModule

    module = CLIPIQAModule({"warning_threshold": 0.4})
    module._ml_available = True
    calls = []

    def metric(batch):
        assert torch.is_inference_mode_enabled()
        calls.append((len(batch), tuple(batch.shape[-2:])))
        scores = batch[:, 0, 0, 0]
        # Model a batched kernel rounding across the validation threshold.
        return torch.where((scores == 0.4) & (len(batch) > 2), scores - 0.00025, scores)

    module._metric = metric
    frames = {
        "a": [np.full((2, 2, 3), 51, dtype=np.uint8)] * 2,
        "b": [np.full((2, 2, 3), 102, dtype=np.uint8)] * 2,
        "c": [np.full((3, 3, 3), 200, dtype=np.uint8)],
    }
    monkeypatch.setattr(module, "_load_frames", lambda sample: frames.get(sample.path.stem, []))
    samples = [Sample(path=f"{name}.mp4", is_video=True) for name in ("a", "b", "c", "missing")]
    single = [module.process(sample.model_copy(deep=True)) for sample in samples]
    calls.clear()

    assert module.process_batch(samples) == single
    assert calls == [(4, (2, 2)), (2, (2, 2)), (1, (3, 3))]
    assert len(samples[0].validation_issues) == 1
    assert not samples[1].validation_issues
