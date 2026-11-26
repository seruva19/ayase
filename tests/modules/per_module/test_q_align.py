"""Tests for q_align module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_q_align_basics():
    from ayase.modules.q_align import QAlignModule
    _test_module_basics(QAlignModule, "q_align")

def test_q_align_image(image_sample):
    from ayase.modules.q_align import QAlignModule
    image_sample.quality_metrics = QualityMetrics()
    m = QAlignModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_q_align_video(video_sample):
    from ayase.modules.q_align import QAlignModule
    video_sample.quality_metrics = QualityMetrics()
    m = QAlignModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

def test_q_align_batches_equal_length_videos_and_replays_warning(monkeypatch, tmp_path):
    import numpy as np
    import pytest

    torch = pytest.importorskip("torch")
    from ayase.models import Sample
    from ayase.modules.q_align import QAlignModule

    module = QAlignModule({"batch_inputs": True})
    module._ml_available = True
    calls = []
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    monkeypatch.setattr(module, "_decode_video", lambda path: [frame] * (3 if path.stem == "c" else 2))

    class Model:
        def score(self, inputs, task_, input_):
            assert input_ == "video"
            calls.append((task_, [len(clip) for clip in inputs]))
            if len(inputs) == 2:
                return torch.tensor([3.5, 2.505])
            return torch.tensor([2.49 if len(inputs[0]) == 2 else 4.0])

    module._model = Model()
    samples = [Sample(path=tmp_path / f"{name}.mp4", is_video=True) for name in "abc"]
    assert module.process_batch(samples) == samples
    assert calls == [("quality", [2, 2]), ("aesthetics", [2, 2]),
                     ("quality", [2]), ("aesthetics", [2]),
                     ("quality", [3]), ("aesthetics", [3])]
    assert [sample.quality_metrics.qalign_quality for sample in samples] == pytest.approx([3.5, 2.49, 4.0])
    assert [len(sample.validation_issues) for sample in samples] == [0, 1, 0]
