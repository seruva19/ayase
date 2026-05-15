"""Tests for audio_peaq module."""

import wave
from pathlib import Path

import numpy as np

from ..conftest import _test_module_basics
from ayase.models import Sample


def _write_wav(path: Path, sr: int, audio: np.ndarray) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def test_audio_peaq_basics():
    from ayase.modules.audio_peaq import AudioPEAQModule
    _test_module_basics(AudioPEAQModule, "audio_peaq")


def test_audio_peaq_lsd_fallback_identical_signal(tmp_dir):
    from ayase.modules.audio_peaq import AudioPEAQModule

    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sig = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    ref_path = tmp_dir / "ref.wav"
    dist_path = tmp_dir / "dist.wav"
    _write_wav(ref_path, sr, sig)
    _write_wav(dist_path, sr, sig)

    m = AudioPEAQModule(config={"backend": "lsd"})
    m.on_mount()
    sample = Sample(path=dist_path, is_video=False, reference_path=ref_path)
    out = m.process(sample)

    assert out.quality_metrics is not None
    assert out.quality_metrics.peaq_odg is not None
    assert out.quality_metrics.peaq_di is not None
    # Identical signals → near-zero distortion → ODG close to 0
    assert out.quality_metrics.peaq_odg >= -0.5, f"identical signals should score near 0, got {out.quality_metrics.peaq_odg}"


def test_audio_peaq_lsd_fallback_degraded_signal(tmp_dir):
    from ayase.modules.audio_peaq import AudioPEAQModule

    sr = 48000
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sig = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    noisy = sig + 0.3 * rng.standard_normal(sig.size).astype(np.float32)
    ref_path = tmp_dir / "ref.wav"
    dist_path = tmp_dir / "noisy.wav"
    _write_wav(ref_path, sr, sig)
    _write_wav(dist_path, sr, noisy)

    m = AudioPEAQModule(config={"backend": "lsd"})
    m.on_mount()
    sample = Sample(path=dist_path, is_video=False, reference_path=ref_path)
    out = m.process(sample)

    assert out.quality_metrics.peaq_odg < -0.1, "noisy signal should score worse than identical"
    assert -4.0 <= out.quality_metrics.peaq_odg <= 0.0


def test_audio_peaq_no_reference_is_noop(tmp_dir):
    from ayase.modules.audio_peaq import AudioPEAQModule

    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sig = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = tmp_dir / "lonely.wav"
    _write_wav(path, sr, sig)

    m = AudioPEAQModule(config={"backend": "lsd"})
    m.on_mount()
    sample = Sample(path=path, is_video=False)  # no reference_path
    out = m.process(sample)
    # No reference → no scoring
    assert out.quality_metrics is None or out.quality_metrics.peaq_odg is None
