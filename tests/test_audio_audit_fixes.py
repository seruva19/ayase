"""Regression tests for audio loading and multichannel resampling."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from ayase.audio import _linear_resample, extract_audio_with_ffmpeg, load_audio


def test_real_stereo_wav_resampling_preserves_duration_and_channels(tmp_path):
    sf = pytest.importorskip("soundfile")
    time = np.arange(48000, dtype=np.float32) / 48000
    audio = np.column_stack([0.5 * np.sin(2 * np.pi * 440 * time), np.zeros_like(time)])
    path = tmp_path / "stereo.wav"
    sf.write(path, audio, 48000)
    result = load_audio(path, target_sr=16000, mono=False)
    assert result.shape == (16000, 2)
    assert np.std(result[:, 0]) > 0.3
    assert np.max(np.abs(result[:, 1])) < 1e-6


def test_linear_resample_preserves_time_first_stereo_channels():
    audio = np.column_stack(
        [
            np.linspace(-1.0, 1.0, 8, dtype=np.float32),
            np.linspace(1.0, -1.0, 8, dtype=np.float32),
        ]
    )

    result = _linear_resample(audio, sr=8, target_sr=16)

    assert result.shape == (16, 2)
    assert result.dtype == np.float32
    assert result[0, 0] == -1.0
    assert result[0, 1] == 1.0
    assert np.allclose(result[:, 0], -result[:, 1])


def test_soundfile_stereo_resamples_on_time_axis(monkeypatch, tmp_path):
    audio = np.column_stack(
        [np.arange(6, dtype=np.float32), 100 + np.arange(6, dtype=np.float32)]
    )
    calls = []

    fake_soundfile = SimpleNamespace(read=lambda *_args, **_kwargs: (audio, 6))

    def fake_resample(values, *, orig_sr, target_sr, axis):
        calls.append((values.shape, orig_sr, target_sr, axis))
        return np.repeat(values, 2, axis=axis)

    fake_librosa = SimpleNamespace(resample=fake_resample)
    monkeypatch.setitem(sys.modules, "soundfile", fake_soundfile)
    monkeypatch.setitem(sys.modules, "librosa", fake_librosa)

    result = load_audio(tmp_path / "stereo.wav", target_sr=12, mono=False)

    assert calls == [((6, 2), 6, 12, 0)]
    assert result.shape == (12, 2)
    assert np.array_equal(result[:, 0], np.repeat(audio[:, 0], 2))
    assert np.array_equal(result[:, 1], np.repeat(audio[:, 1], 2))


def test_soundfile_stereo_uses_multichannel_linear_fallback(monkeypatch, tmp_path):
    audio = np.column_stack(
        [np.linspace(0, 1, 4, dtype=np.float32), np.linspace(1, 0, 4, dtype=np.float32)]
    )
    fake_soundfile = SimpleNamespace(read=lambda *_args, **_kwargs: (audio, 4))
    def unavailable_resampler(*_args, **_kwargs):
        raise RuntimeError("resampler unavailable")

    fake_librosa = SimpleNamespace(resample=unavailable_resampler)
    monkeypatch.setitem(sys.modules, "soundfile", fake_soundfile)
    monkeypatch.setitem(sys.modules, "librosa", fake_librosa)

    result = load_audio(tmp_path / "stereo.wav", target_sr=8, mono=False)

    assert result.shape == (8, 2)
    assert np.allclose(result[:, 0], 1.0 - result[:, 1])


def test_librosa_fallback_converts_channels_first_to_time_first(monkeypatch, tmp_path):
    channels_first = np.vstack(
        [np.arange(5, dtype=np.float32), 10 + np.arange(5, dtype=np.float32)]
    )
    fake_soundfile = SimpleNamespace(
        read=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("unsupported"))
    )
    fake_librosa = SimpleNamespace(load=lambda *_args, **_kwargs: (channels_first, 16000))
    monkeypatch.setitem(sys.modules, "soundfile", fake_soundfile)
    monkeypatch.setitem(sys.modules, "librosa", fake_librosa)

    result = load_audio(tmp_path / "container.mp4", target_sr=16000, mono=False)

    assert result.shape == (5, 2)
    assert np.array_equal(result[:, 0], channels_first[0])
    assert np.array_equal(result[:, 1], channels_first[1])


def test_ffmpeg_stereo_output_remains_time_first(monkeypatch, tmp_path):
    audio = np.column_stack(
        [np.arange(7, dtype=np.float32), 20 + np.arange(7, dtype=np.float32)]
    )
    fake_soundfile = SimpleNamespace(read=lambda *_args, **_kwargs: (audio, 16000))
    monkeypatch.setitem(sys.modules, "soundfile", fake_soundfile)
    monkeypatch.setattr(
        "ayase.audio.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
    )

    result = extract_audio_with_ffmpeg(
        tmp_path / "container.mp4", target_sr=16000, mono=False
    )

    assert result.shape == (7, 2)
    assert np.array_equal(result, audio)
