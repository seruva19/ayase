"""UTMOSv2 module tests.

The published API is ``utmosv2.create_model(pretrained=True)`` followed by
``model.predict(data=<waveform>, sr=<rate>)`` (or ``predict(input_path=...)``).
The package exposes no module-level ``predict``/``score`` functions, so the
module must instantiate the model and call the official keyword API.
"""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from ayase.models import QualityMetrics, Sample
from ayase.modules.audio_utmos_v2 import AudioUTMOSv2Module


class _FakeUTMOSModel:
    """Mimics the model returned by ``utmosv2.create_model``."""

    def __init__(self, mos=3.7):
        self.mos = mos
        self.calls = []

    def predict(self, *, data=None, sr=None, input_path=None):
        self.calls.append({"data": data, "sr": sr, "input_path": input_path})
        return np.asarray([self.mos], dtype=np.float32)


def _fake_package(model):
    pkg = SimpleNamespace(create_model=lambda pretrained=True: model)
    pkg.__name__ = "utmosv2"
    return pkg


def _video_sample(tmp_path):
    return Sample(path=tmp_path / "clip.mp4", is_video=True)


def test_setup_uses_create_model(monkeypatch, tmp_path):
    """setup() must build the official inference model, not keep the package."""
    model = _FakeUTMOSModel()
    monkeypatch.setitem(sys.modules, "utmosv2", _fake_package(model))
    module = AudioUTMOSv2Module()
    module.setup()
    assert module._backend == "utmosv2_package"
    assert module._model is model


def test_predict_official_keyword_api(monkeypatch, tmp_path):
    """Scoring must use ``predict(data=..., sr=...)`` as published."""
    model = _FakeUTMOSModel(mos=4.25)
    module = AudioUTMOSv2Module()
    module._backend = "utmosv2_package"
    module._model = model
    monkeypatch.setattr(
        "ayase.modules.audio_utmos_v2.load_audio",
        lambda *a, **k: np.zeros(16000, dtype=np.float32),
    )
    sample = module.process(_video_sample(tmp_path))
    assert model.calls, "predict was never called"
    assert model.calls[0]["sr"] == 16000
    assert model.calls[0]["data"] is not None
    assert sample.quality_metrics is not None
    assert sample.quality_metrics.utmos_v2_score == pytest.approx(4.25, abs=1e-3)


def test_no_model_no_score(monkeypatch, tmp_path):
    """Unavailable backend must leave the field unset, never fabricate."""
    module = AudioUTMOSv2Module()
    module._backend = "unavailable"
    sample = module.process(_video_sample(tmp_path))
    if sample.quality_metrics is not None:
        assert sample.quality_metrics.utmos_v2_score is None
