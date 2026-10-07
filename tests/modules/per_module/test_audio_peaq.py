"""PEAQ BASIC via peaqb-fast module tests.

If the peaqb output yields only one of ODG/DI, the missing value must stay
``None`` — substituting ``0.0`` fabricates a published-metric number.
"""

from pathlib import Path
from types import SimpleNamespace

from ayase.modules.audio_peaq import AudioPEAQModule, _parse_peaqb_output


def test_parse_both_values():
    out = _parse_peaqb_output("Objective Difference Grade: ODG = -1.234\nDistortion Index: DI = 5.6\n")
    assert out == (-1.234, 5.6)


def test_missing_di_stays_none():
    out = _parse_peaqb_output("ODG: -2.5\n")
    assert out is not None
    assert out[0] == -2.5
    assert out[1] is None


def test_missing_odg_stays_none():
    out = _parse_peaqb_output("Distortion Index: DI = 3.1\n")
    assert out is not None
    assert out[0] is None
    assert out[1] == 3.1


def test_no_values_returns_none():
    assert _parse_peaqb_output("peaqb: error: cannot decode input") is None


def test_peaqb_uses_upstream_reference_and_test_flags(monkeypatch, tmp_path):
    module = AudioPEAQModule()
    module._peaqb_path = "peaqb"
    wavs = iter((tmp_path / "reference.wav", tmp_path / "test.wav"))
    for path in (tmp_path / "reference.wav", tmp_path / "test.wav"):
        path.touch()
    monkeypatch.setattr(module, "_to_wav", lambda _path: next(wavs))
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=0, stdout="DI: 2.5\nODG: -1.0", stderr="")

    monkeypatch.setattr("ayase.modules.audio_peaq.subprocess.run", fake_run)

    assert module._run_peaqb(Path("ref.flac"), Path("test.flac")) == (-1.0, 2.5)
    assert calls == [
        (
            ["peaqb", "-r", str(tmp_path / "reference.wav"), "-t", str(tmp_path / "test.wav")],
            {"capture_output": True, "text": True, "timeout": 120},
        )
    ]


def test_nonzero_peaqb_exit_ignores_partial_scores(monkeypatch, tmp_path):
    module = AudioPEAQModule()
    module._peaqb_path = "peaqb"
    wavs = iter((tmp_path / "reference.wav", tmp_path / "test.wav"))
    for path in (tmp_path / "reference.wav", tmp_path / "test.wav"):
        path.touch()
    monkeypatch.setattr(module, "_to_wav", lambda _path: next(wavs))
    monkeypatch.setattr(
        "ayase.modules.audio_peaq.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=2, stdout="ODG: -0.5\nDI: 4.0", stderr="decode failed"
        ),
    )

    assert module._run_peaqb(Path("ref.flac"), Path("test.flac")) is None


def test_advanced_mode_stays_unavailable(monkeypatch):
    module = AudioPEAQModule({"mode": "advanced"})
    monkeypatch.setattr("ayase.modules.audio_peaq.shutil.which", lambda _name: "peaqb")

    module.setup()

    assert module.active_backend == "unavailable"
    assert module._backend == "unavailable"
    assert module._peaqb_path is None


def test_advanced_mode_cannot_invoke_basic_binary(monkeypatch):
    module = AudioPEAQModule({"mode": "advanced"})
    module._peaqb_path = "peaqb"
    monkeypatch.setattr(
        "ayase.modules.audio_peaq.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not execute")),
    )

    assert module._run_peaqb(Path("ref.wav"), Path("test.wav")) is None


def test_peaqb_metadata_is_adapted():
    assert AudioPEAQModule.provenance == "adapted"
    assert all("BASIC" in source for source in AudioPEAQModule.sources.values())
    assert all("48 kHz mono" in text for text in AudioPEAQModule.deviations.values())
