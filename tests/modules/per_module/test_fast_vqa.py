"""Tests for fast_vqa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_fast_vqa_basics():
    from ayase.modules.fast_vqa import FastVQAModule
    _test_module_basics(FastVQAModule, "fast_vqa")

def test_fast_vqa_video(video_sample):
    from ayase.modules.fast_vqa import FastVQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = FastVQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample
def test_fast_vqa_batches_fragments_and_reuses_prepared_inputs(monkeypatch, tmp_path):
    import math

    import pytest

    torch = pytest.importorskip("torch")
    from ayase.models import Sample
    from ayase.modules.fast_vqa import FastVQAModule

    assert not FastVQAModule().batch_fragments
    module = FastVQAModule({"batch_fragments": True})
    module._ml_available = True
    mean, std = module.fastvqa_mean_std[module.model_type]
    boundary = mean + std * math.log(2 / 3)
    values = {"clear": [0.4, 0.3], "threshold": [boundary], "failure": [0.18]}
    preparations = []
    calls = []

    def prepare(path):
        preparations.append(path.stem)
        return {"samples": {"fragments": torch.tensor(values[path.stem]).view(-1, 1)}}

    class Evaluator:
        reject = False

        def __call__(self, inputs):
            features = inputs["fragments"]
            calls.append(len(features))
            if self.reject and len(features) > 2:
                raise RuntimeError("batch failed")
            return features

    evaluator = Evaluator()
    monkeypatch.setattr(module, "_prepare_input", prepare)
    monkeypatch.setattr(module, "_model", evaluator)

    def samples():
        return [Sample(path=tmp_path / f"{name}.mp4", is_video=True) for name in values]

    expected = [module.process(sample).model_dump() for sample in samples()]
    for reject, sizes in ((False, [4, 1]), (True, [4, 2, 1, 1])):
        evaluator.reject = reject
        preparations.clear()
        calls.clear()
        output = module.process_batch(samples())
        assert preparations == list(values)
        assert calls == sizes
        assert [sample.model_dump() for sample in output] == expected
