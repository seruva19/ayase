"""Managed ViSQOL backend behavior."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from ayase.modules.visqol import ViSQOLModule


class _PortApi:
    instances = []

    def __init__(self) -> None:
        self.created = None
        self.measured = None
        self.instances.append(self)

    def create(self, **kwargs) -> None:
        self.created = kwargs

    def measure(self, reference: str, degraded: str):
        self.measured = (reference, degraded)
        return SimpleNamespace(moslqo=4.125)


def _port_module(api_class=_PortApi) -> ModuleType:
    package = ModuleType("visqol")
    package.VisqolApi = api_class
    return package


def test_port_audio_backend_identity_and_score(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "visqol", _port_module())
    module = ViSQOLModule({"mode": "audio", "_ayase_managed_worker": True})

    module.setup()

    assert module._backend == "visqol_python_port"
    assert module._ml_available is True
    assert module._visqol_api.created == {"mode": "audio"}
    assert module._compute_visqol_port("reference.wav", "degraded.wav") == 4.125


def test_speech_port_requires_lattice(monkeypatch: pytest.MonkeyPatch) -> None:
    class LatticeApi(_PortApi):
        def create(self, **kwargs) -> None:
            super().create(**kwargs)
            if kwargs.get("use_lattice_model") is not True:
                raise AssertionError("speech must explicitly require lattice")

    monkeypatch.setitem(sys.modules, "visqol", _port_module(LatticeApi))
    module = ViSQOLModule({"mode": "speech", "_ayase_managed_worker": True})

    module.setup()

    assert module._backend == "visqol_python_port"
    assert module._visqol_api.created == {"mode": "speech", "use_lattice_model": True}


def test_missing_lattice_never_uses_polynomial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MissingLatticeApi:
        def create(self, **kwargs) -> None:
            assert kwargs == {"mode": "speech", "use_lattice_model": True}
            raise RuntimeError("lattice runtime missing")

    monkeypatch.setitem(sys.modules, "visqol", _port_module(MissingLatticeApi))
    monkeypatch.setattr(
        "ayase.modules.visqol.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError()),
    )
    module = ViSQOLModule({"mode": "speech", "_ayase_managed_worker": True})

    module.setup()

    assert module._backend == "unavailable"
    assert module._ml_available is False


def test_process_uses_port_paths_and_preserves_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "visqol", _port_module())
    reference = tmp_path / "reference.wav"
    degraded = tmp_path / "degraded.wav"
    reference.touch()
    degraded.touch()
    from ayase.models import Sample

    sample = Sample(path=degraded, reference_path=reference, is_video=True)
    module = ViSQOLModule({"_ayase_managed_worker": True})
    module.setup()

    result = module.process(sample)

    assert result is sample
    assert result.quality_metrics is not None
    assert result.quality_metrics.visqol == 4.125
    assert module._visqol_api.measured == (str(reference), str(degraded))
