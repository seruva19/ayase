"""Tests for the official Distill-MOS speech-quality module."""

import sys
import types

import numpy as np
import pytest

from ayase.models import Sample
from tests.modules.conftest import _test_module_basics


def test_audio_distill_mos_basics_and_public_export():
    from ayase.modules import AudioDistillMOSModule

    _test_module_basics(AudioDistillMOSModule, "audio_distill_mos")
    assert AudioDistillMOSModule.required_packages == ["distillmos"]
    assert AudioDistillMOSModule.metric_groups == {"distill_mos_score": "audio"}


def test_audio_distill_mos_setup_uses_fixed_window_official_model(monkeypatch):
    import ayase.runtime as runtime
    from ayase.modules.audio_distill_mos import AudioDistillMOSModule

    created = {}

    class FakeModel:
        def __init__(self, *, segmenting_in_forward):
            created["segmenting_in_forward"] = segmenting_in_forward
            self.device = None
            self.eval_called = False

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.eval_called = True
            return self

    fake_distillmos = types.ModuleType("distillmos")
    fake_distillmos.ConvTransformerSQAModel = FakeModel
    monkeypatch.setitem(sys.modules, "distillmos", fake_distillmos)
    monkeypatch.setattr(runtime, "resolve_torch_device", lambda _device: "cpu")

    module = AudioDistillMOSModule({"device": "cpu"})
    module.setup()

    assert created == {"segmenting_in_forward": False}
    assert module._model.device == "cpu"
    assert module._model.eval_called
    assert module._backend == "distillmos:0.9.1/v7"


def test_audio_distill_mos_short_input_is_right_padded():
    from ayase.modules.audio_distill_mos import AudioDistillMOSModule

    module = AudioDistillMOSModule()
    audio = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)

    windows = module._select_windows(audio)

    assert len(windows) == 1
    assert windows[0].shape == (122880,)
    np.testing.assert_array_equal(windows[0][: audio.size], audio)
    assert np.count_nonzero(windows[0][audio.size :]) == 0


def test_audio_distill_mos_reproduces_official_window_starts_under_cap():
    from ayase.modules.audio_distill_mos import AudioDistillMOSModule

    module = AudioDistillMOSModule({"max_windows": 12})
    audio = np.arange(160000, dtype=np.float32) + 1.0

    windows = module._select_windows(audio)

    # overlength=37120; official count=ceil(37120/16000)+1=4
    assert len(windows) == 4
    assert [window[0] for window in windows] == pytest.approx(
        [audio[0], audio[12373], audio[24746], audio[37120]]
    )
    assert windows[-1][-1] == pytest.approx(audio[-1])


def test_audio_distill_mos_caps_long_input_and_includes_tail():
    from ayase.modules.audio_distill_mos import AudioDistillMOSModule

    module = AudioDistillMOSModule({"max_windows": 3})
    audio = np.arange(30 * 16000, dtype=np.float32) + 1.0

    windows = module._select_windows(audio)

    assert len(windows) == 3
    assert windows[0][0] == pytest.approx(audio[0])
    assert windows[-1][-1] == pytest.approx(audio[-1])


def test_audio_distill_mos_single_window_cap_is_tail_aligned():
    from ayase.modules.audio_distill_mos import AudioDistillMOSModule

    module = AudioDistillMOSModule({"max_windows": 1})
    audio = np.arange(10 * 16000, dtype=np.float32) + 1.0

    windows = module._select_windows(audio)

    assert len(windows) == 1
    assert windows[0][0] == pytest.approx(audio[-122880])
    assert windows[0][-1] == pytest.approx(audio[-1])


def test_audio_distill_mos_process_uses_first_channel_and_averages(monkeypatch, tmp_path):
    import torch
    import ayase.modules.audio_distill_mos as distill_module

    captured = {}

    class FakeModel:
        def __call__(self, tensor):
            captured["tensor"] = tensor.detach().cpu().numpy()
            return torch.tensor([[2.0]], dtype=torch.float32)

    left = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)
    right = np.ones(16000, dtype=np.float32)
    stereo = np.column_stack([left, right])
    load_call = {}

    def fake_load(path, target_sr, mono):
        load_call.update(path=path, target_sr=target_sr, mono=mono)
        return stereo

    monkeypatch.setattr(distill_module, "load_audio", fake_load)
    module = distill_module.AudioDistillMOSModule()
    module._model = FakeModel()
    module._backend = "distillmos:0.9.1/v7"
    sample = Sample(path=tmp_path / "speech.wav", is_video=False)

    result = module.process(sample)

    assert result is sample
    assert result.quality_metrics is not None
    assert result.quality_metrics.distill_mos_score == pytest.approx(2.0)
    assert load_call == {"path": sample.path, "target_sr": 16000, "mono": False}
    np.testing.assert_allclose(captured["tensor"][0, : left.size], left)
    assert np.count_nonzero(captured["tensor"][0, left.size :]) == 0


def test_audio_distill_mos_process_means_window_scores(monkeypatch, tmp_path):
    import torch
    import ayase.modules.audio_distill_mos as distill_module

    class FakeModel:
        def __call__(self, tensor):
            assert tensor.shape[0] == 3
            return torch.tensor([[2.0], [3.0], [4.0]], dtype=torch.float32)

    audio = np.ones(30 * 16000, dtype=np.float32) * 0.1
    monkeypatch.setattr(distill_module, "load_audio", lambda *args, **kwargs: audio)
    module = distill_module.AudioDistillMOSModule({"max_windows": 3})
    module._model = FakeModel()
    sample = Sample(path=tmp_path / "long.wav", is_video=False)

    result = module.process(sample)

    assert result.quality_metrics is not None
    assert result.quality_metrics.distill_mos_score == pytest.approx(3.0)


@pytest.mark.parametrize(
    "audio",
    [
        None,
        np.zeros(2 * 16000, dtype=np.float32),
        np.ones(8000, dtype=np.float32) * 0.1,
    ],
)
def test_audio_distill_mos_missing_silent_or_short_is_noop(
    monkeypatch, tmp_path, audio
):
    import ayase.modules.audio_distill_mos as distill_module

    module = distill_module.AudioDistillMOSModule()
    module._model = object()
    monkeypatch.setattr(distill_module, "load_audio", lambda *args, **kwargs: audio)
    sample = Sample(path=tmp_path / "speech.wav", is_video=False)

    result = module.process(sample)

    assert result is sample
    assert result.quality_metrics is None


def test_audio_distill_mos_image_is_noop(monkeypatch, image_sample):
    import ayase.modules.audio_distill_mos as distill_module

    module = distill_module.AudioDistillMOSModule()
    module._model = object()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("image inputs must not be decoded as audio")

    monkeypatch.setattr(distill_module, "load_audio", fail_if_called)

    result = module.process(image_sample)

    assert result is image_sample
    assert result.quality_metrics is None


@pytest.mark.parametrize("scores", [[float("nan")], [0.99], [5.01]])
def test_audio_distill_mos_rejects_invalid_scores(monkeypatch, tmp_path, scores):
    import torch
    import ayase.modules.audio_distill_mos as distill_module

    class FakeModel:
        def __call__(self, _tensor):
            return torch.tensor(scores, dtype=torch.float32)

    monkeypatch.setattr(
        distill_module,
        "load_audio",
        lambda *args, **kwargs: np.ones(16000, dtype=np.float32) * 0.1,
    )
    module = distill_module.AudioDistillMOSModule()
    module._model = FakeModel()
    sample = Sample(path=tmp_path / "speech.wav", is_video=False)

    result = module.process(sample)

    assert result is sample
    assert result.quality_metrics is None


def test_audio_distill_mos_failure_returns_same_sample(monkeypatch, tmp_path):
    import ayase.modules.audio_distill_mos as distill_module

    monkeypatch.setattr(
        distill_module,
        "load_audio",
        lambda *args, **kwargs: np.ones(16000, dtype=np.float32) * 0.1,
    )
    module = distill_module.AudioDistillMOSModule()
    module._model = lambda _tensor: (_ for _ in ()).throw(RuntimeError("failure"))
    sample = Sample(path=tmp_path / "speech.wav", is_video=False)

    result = module.process(sample)

    assert result is sample
    assert result.quality_metrics is None


def test_audio_distill_mos_warning_threshold(monkeypatch, tmp_path):
    import torch
    import ayase.modules.audio_distill_mos as distill_module

    class FakeModel:
        def __call__(self, _tensor):
            return torch.tensor([2.5], dtype=torch.float32)

    monkeypatch.setattr(
        distill_module,
        "load_audio",
        lambda *args, **kwargs: np.ones(16000, dtype=np.float32) * 0.1,
    )
    module = distill_module.AudioDistillMOSModule({"warning_threshold": 3.0})
    module._model = FakeModel()
    sample = Sample(path=tmp_path / "speech.wav", is_video=False)

    result = module.process(sample)

    assert result.quality_metrics.distill_mos_score == pytest.approx(2.5)
    assert len(result.validation_issues) == 1
    assert result.validation_issues[0].details == {"distill_mos_score": 2.5}


def test_audio_distill_mos_checkpoint_metadata_is_exact():
    from ayase.modules.audio_distill_mos import AudioDistillMOSModule

    model = AudioDistillMOSModule.models[0]
    assert model["id"] == "microsoft/Distill-MOS:distill_mos_v7.pt"
    assert model["type"] == "pip_package"
    assert model["install"] == "pip install distillmos==0.9.1"
    assert model["url"] == (
        "https://raw.githubusercontent.com/microsoft/Distill-MOS/"
        "b8d46ee2748176155619cda5315ab4d5ef6af28d/"
        "distillmos/weights/distill_mos_v7.pt"
    )
    assert model["size"] == "16,907,522 bytes (16.124269 MiB)"
    assert model["license"] == "MIT"
    assert "b18b3ac60227267cfb91e5d00ce22cc7b73716fd92f269484da1446e27031a40" in model[
        "notes"
    ]
