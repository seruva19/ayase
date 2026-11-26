"""Tests for action_recognition module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_action_recognition_basics():
    from ayase.modules.action_recognition import ActionRecognitionModule
    _test_module_basics(ActionRecognitionModule, "action_recognition")

def test_action_recognition_video(video_sample):
    from ayase.modules.action_recognition import ActionRecognitionModule
    video_sample.quality_metrics = QualityMetrics()
    m = ActionRecognitionModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample
def test_action_recognition_batches_and_retries_threshold_samples(monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    import pytest

    torch = pytest.importorskip("torch")
    from ayase.models import Sample
    from ayase.modules.action_recognition import ActionRecognitionModule

    module = ActionRecognitionModule({"top_k": 2})
    module._ml_available = True
    module._device = "cpu"
    batch_sizes = []

    class Inputs(dict):
        def to(self, device):
            return self

    class Classifier:
        config = SimpleNamespace(id2label={0: "walk", 1: "sit", 2: "stand"})
        reject_batches = False

        def __call__(self, pixel_values):
            batch_sizes.append(len(pixel_values))
            if self.reject_batches and len(pixel_values) > 1:
                raise RuntimeError("Batch execution failed")
            return SimpleNamespace(logits=pixel_values)

    monkeypatch.setattr(module, "_model", Classifier())
    monkeypatch.setattr(
        module, "_load_frames", lambda sample, num_frames: [sample.metadata["logits"]] * 16
    )
    monkeypatch.setattr(
        module, "_processor",
        lambda frames, return_tensors: Inputs(pixel_values=torch.tensor([frames[0]])),
    )
    inputs = [
        Sample(path=Path("clear.mp4"), is_video=True, metadata={"logits": [6.0, 1.0, 1.0]}),
        Sample(path=Path("boundary.mp4"), is_video=True, metadata={"logits": [0.0, 0.0, -100.0]}),
        Sample(path=Path("image.png"), is_video=False),
    ]
    expected = [module.process(sample.model_copy(deep=True)).model_dump() for sample in inputs]
    for reject_batches, sizes in ((False, [2, 1]), (True, [2, 1, 1])):
        module._model.reject_batches = reject_batches
        batch_sizes.clear()
        samples = [sample.model_copy(deep=True) for sample in inputs]
        results = module.process_batch(samples)
        assert len(results) == len(samples)
        assert all(result is sample for result, sample in zip(results, samples))
        assert batch_sizes == sizes
        assert [item.model_dump() for item in results] == expected
