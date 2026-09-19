"""Tests for the optional CDPAM full-reference audio metric."""

import sys
from types import SimpleNamespace

import numpy as np

from ayase.models import Sample

from ..conftest import _test_module_basics


class _FakeDistance:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def item(self):
        return self.value


class _FakeCDPAM:
    def __init__(self, dev, distance=0.4):
        self.device = dev
        self.distance = distance
        self.forward_calls = []

    def forward(self, reference, degraded):
        self.forward_calls.append((reference, degraded))
        return _FakeDistance(self.distance)


def test_cdpam_basics():
    from ayase.modules.cdpam import CDPAMModule

    _test_module_basics(CDPAMModule, "cdpam")
    assert CDPAMModule.metric_groups == {"cdpam_score": "audio"}


def test_cdpam_setup_uses_official_package_api(monkeypatch):
    from ayase.modules.cdpam import CDPAMModule

    created = []

    def create_model(dev):
        model = _FakeCDPAM(dev)
        created.append(model)
        return model

    fake_package = SimpleNamespace(CDPAM=create_model)
    monkeypatch.setitem(sys.modules, "cdpam", fake_package)

    module = CDPAMModule({"device": "cpu"})
    module.setup()

    assert module._backend == "cdpam:0.0.6"
    assert created[0].device == "cpu"


def test_cdpam_routes_reference_then_degraded_and_warns(tmp_path, monkeypatch):
    from ayase.modules.cdpam import CDPAMModule

    reference = tmp_path / "reference.wav"
    degraded = tmp_path / "degraded.wav"
    reference.touch()
    degraded.touch()
    prepared = []

    def prepare_audio(path):
        prepared.append(path)
        return np.array([[1.0]], dtype=np.float32)

    model = _FakeCDPAM("cpu", distance=0.4)
    module = CDPAMModule({"warning_threshold": 0.25})
    module._model = model
    monkeypatch.setattr(module, "_prepare_audio", prepare_audio)
    module._backend = "cdpam:0.0.6"
    sample = Sample(path=degraded, is_video=False, reference_path=reference)

    result = module.process(sample)

    assert result is sample
    assert prepared == [reference, degraded]
    assert len(model.forward_calls) == 1
    assert sample.quality_metrics is not None
    assert sample.quality_metrics.cdpam_score == 0.4
    assert sample.quality_metrics.metric_backends["cdpam_score"] == "cdpam:0.0.6"
    assert len(sample.validation_issues) == 1
    assert sample.validation_issues[0].issue_type == "high_cdpam_distance"


def test_cdpam_below_threshold_does_not_warn(tmp_path, monkeypatch):
    from ayase.modules.cdpam import CDPAMModule

    reference = tmp_path / "reference.wav"
    degraded = tmp_path / "degraded.wav"
    reference.touch()
    degraded.touch()

    module = CDPAMModule({"warning_threshold": 0.25})
    module._model = _FakeCDPAM("cpu", distance=0.0)
    monkeypatch.setattr(
        module, "_prepare_audio", lambda path: np.zeros((1, 16), dtype=np.float32)
    )
    sample = Sample(path=degraded, is_video=False, reference_path=reference)

    result = module.process(sample)

    assert result is sample
    assert sample.quality_metrics.cdpam_score == 0.0
    assert sample.validation_issues == []


def test_cdpam_without_reference_skips_gracefully(tmp_path):
    from ayase.modules.cdpam import CDPAMModule

    degraded = tmp_path / "degraded.wav"
    degraded.touch()
    module = CDPAMModule()
    model = _FakeCDPAM("cpu")
    module._model = model
    sample = Sample(path=degraded, is_video=False)

    result = module.process(sample)

    assert result is sample
    assert sample.quality_metrics is None
    assert model.forward_calls == []


def test_cdpam_preprocessing_matches_official_scale(monkeypatch):
    from ayase.modules import cdpam as module_under_test
    from ayase.modules.cdpam import CDPAMModule

    waveform = np.array([-1.2, -0.5, 0.0, 0.5, 1.2], dtype=np.float32)
    calls = []

    def fake_load(path, target_sr, mono):
        calls.append((path, target_sr, mono))
        return waveform

    monkeypatch.setattr(module_under_test, "load_audio", fake_load)
    result = CDPAMModule()._prepare_audio(module_under_test.Path("audio.wav"))

    assert calls == [(module_under_test.Path("audio.wav"), 22050, True)]
    assert result.shape == (1, 5)
    np.testing.assert_array_equal(
        result,
        np.array([[-32768.0, -16384.0, 0.0, 16384.0, 32768.0]], dtype=np.float32),
    )
