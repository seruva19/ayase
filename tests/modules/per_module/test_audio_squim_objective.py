"""Tests for the TorchAudio-SQUIM Objective module."""

import sys
import types

import numpy as np
import pytest

from ayase.models import Sample
from tests.modules.conftest import _test_module_basics


def test_audio_squim_objective_basics():
    from ayase.modules.audio_squim_objective import AudioSQUIMObjectiveModule

    _test_module_basics(AudioSQUIMObjectiveModule, "audio_squim_objective")
    assert set(AudioSQUIMObjectiveModule.metric_groups) == {
        "squim_stoi_score",
        "squim_pesq_score",
        "squim_si_sdr_score",
    }


def test_audio_squim_objective_setup_uses_public_bundle(monkeypatch):
    from ayase.modules.audio_squim_objective import AudioSQUIMObjectiveModule

    class FakeModel:
        def __init__(self):
            self.device = None
            self.eval_called = False

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.eval_called = True
            return self

    class FakeBundle:
        sample_rate = 16000

        def __init__(self):
            self.model = FakeModel()
            self.get_model_calls = 0

        def get_model(self):
            self.get_model_calls += 1
            return self.model

    bundle = FakeBundle()
    fake_torchaudio = types.ModuleType("torchaudio")
    fake_torchaudio.__path__ = []
    fake_pipelines = types.ModuleType("torchaudio.pipelines")
    fake_pipelines.SQUIM_OBJECTIVE = bundle
    monkeypatch.setitem(sys.modules, "torchaudio", fake_torchaudio)
    monkeypatch.setitem(sys.modules, "torchaudio.pipelines", fake_pipelines)

    module = AudioSQUIMObjectiveModule({"device": "cpu"})
    module.setup()

    assert bundle.get_model_calls == 1
    assert bundle.model.device == "cpu"
    assert bundle.model.eval_called
    assert module._backend == "torchaudio:SQUIM_OBJECTIVE"


def test_audio_squim_objective_windows_are_bounded_and_tail_aligned():
    from ayase.modules.audio_squim_objective import AudioSQUIMObjectiveModule

    sample_rate = 16000
    audio = np.arange(11 * sample_rate, dtype=np.float32) + 1.0
    module = AudioSQUIMObjectiveModule({"max_windows": 3})

    windows = module._select_windows(audio)

    assert len(windows) == 3
    assert all(window.shape == (5 * sample_rate,) for window in windows)
    assert windows[0][0] == pytest.approx(audio[0])
    assert windows[1][0] == pytest.approx(audio[3 * sample_rate])
    assert windows[2][0] == pytest.approx(audio[6 * sample_rate])
    assert windows[-1][-1] == pytest.approx(audio[-1])


def test_audio_squim_objective_single_window_is_tail_aligned():
    from ayase.modules.audio_squim_objective import AudioSQUIMObjectiveModule

    sample_rate = 16000
    audio = np.arange(7 * sample_rate, dtype=np.float32) + 1.0
    module = AudioSQUIMObjectiveModule({"max_windows": 1})

    windows = module._select_windows(audio)

    assert len(windows) == 1
    assert windows[0][0] == pytest.approx(audio[2 * sample_rate])
    assert windows[0][-1] == pytest.approx(audio[-1])


def test_audio_squim_objective_process_maps_and_averages_outputs(monkeypatch, tmp_path):
    import ayase.modules.audio_squim_objective as squim_module

    module = squim_module.AudioSQUIMObjectiveModule({"max_windows": 3})
    module._model = object()
    module._backend = "torchaudio:SQUIM_OBJECTIVE"
    audio = np.ones(11 * 16000, dtype=np.float32) * 0.1
    load_call = {}

    def fake_load_audio(path, target_sr, mono):
        load_call.update(path=path, target_sr=target_sr, mono=mono)
        return audio

    monkeypatch.setattr(squim_module, "load_audio", fake_load_audio)

    returned = iter(((0.70, 2.0, -2.0), (0.80, 3.0, 0.0), (0.90, 4.0, 5.0)))
    monkeypatch.setattr(module, "_score_window", lambda _window: next(returned))
    sample = Sample(path=tmp_path / "speech.wav", is_video=False)

    result = module.process(sample)

    assert result is sample
    assert result.quality_metrics is not None
    assert result.quality_metrics.squim_stoi_score == pytest.approx(0.8)
    assert result.quality_metrics.squim_pesq_score == pytest.approx(3.0)
    assert result.quality_metrics.squim_si_sdr_score == pytest.approx(1.0)
    assert load_call == {"path": sample.path, "target_sr": 16000, "mono": True}


@pytest.mark.parametrize(
    "audio",
    [
        None,
        np.zeros(2 * 16000, dtype=np.float32),
        np.ones(1000, dtype=np.float32),
    ],
)
def test_audio_squim_objective_missing_silent_or_short_is_noop(
    monkeypatch, tmp_path, audio
):
    import ayase.modules.audio_squim_objective as squim_module

    module = squim_module.AudioSQUIMObjectiveModule()
    module._model = object()
    monkeypatch.setattr(squim_module, "load_audio", lambda *args, **kwargs: audio)
    sample = Sample(path=tmp_path / "speech.wav", is_video=False)

    result = module.process(sample)

    assert result is sample
    assert result.quality_metrics is None


def test_audio_squim_objective_image_is_noop(monkeypatch, image_sample):
    import ayase.modules.audio_squim_objective as squim_module

    module = squim_module.AudioSQUIMObjectiveModule()
    module._model = object()
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("load_audio should not probe image inputs")

    monkeypatch.setattr(squim_module, "load_audio", fail_if_called)
    result = module.process(image_sample)

    assert result is image_sample
    assert not called


def test_audio_squim_objective_failure_returns_same_sample(monkeypatch, tmp_path):
    import ayase.modules.audio_squim_objective as squim_module

    module = squim_module.AudioSQUIMObjectiveModule()
    module._model = object()
    monkeypatch.setattr(
        squim_module,
        "load_audio",
        lambda *args, **kwargs: np.ones(16000, dtype=np.float32),
    )
    monkeypatch.setattr(
        module,
        "_score_window",
        lambda _window: (_ for _ in ()).throw(RuntimeError("test failure")),
    )
    sample = Sample(path=tmp_path / "speech.wav", is_video=False)

    result = module.process(sample)

    assert result is sample
    assert result.quality_metrics is None


def test_audio_squim_objective_checkpoint_metadata_is_exact():
    from ayase.modules.audio_squim_objective import AudioSQUIMObjectiveModule

    model = AudioSQUIMObjectiveModule.models[0]
    assert model["id"] == "squim_objective_dns2020.pth"
    assert model["type"] == "local"
    assert model["url"] == (
        "https://download.pytorch.org/torchaudio/models/squim_objective_dns2020.pth"
    )
    assert model["size"] == "29,584,237 bytes (28.213727 MiB)"
    assert model["license"] == "CC BY 4.0"
    assert "2c54586fea83fb5eb5394d710038ee89f55cab7011a5bf730bebed4c8777e828" in model[
        "notes"
    ]
