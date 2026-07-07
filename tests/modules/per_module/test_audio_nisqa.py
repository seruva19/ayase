"""Tests for audio_nisqa module."""

import wave
from pathlib import Path

import numpy as np
import pytest

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics, Sample


def _write_synthetic_wav(tmp_dir: Path, sr: int = 16000, seconds: float = 1.0) -> Path:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    # Mix of speech-like sinusoids + small noise
    wave_arr = (0.5 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 440 * t)
                + 0.05 * np.random.randn(t.size)).astype(np.float32)
    pcm = (wave_arr * 32767).astype(np.int16)
    path = tmp_dir / "speech.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return path


def test_audio_nisqa_basics():
    from ayase.modules.audio_nisqa import AudioNISQAModule
    _test_module_basics(AudioNISQAModule, "audio_nisqa")


def test_audio_nisqa_real_backend_or_none(tmp_dir):
    """No spectral proxy tier: real NISQA scores or all fields left unset."""
    from ayase.modules.audio_nisqa import AudioNISQAModule
    wav_path = _write_synthetic_wav(tmp_dir)
    sample = Sample(path=wav_path, is_video=False)

    m = AudioNISQAModule()
    m.on_mount()
    out = m.process(sample)

    fields = ("nisqa_mos", "nisqa_noisiness", "nisqa_coloration",
              "nisqa_discontinuity", "nisqa_loudness")
    if m.active_backend == "nisqa":
        assert out.quality_metrics is not None
        for field in fields:
            v = getattr(out.quality_metrics, field)
            assert v is not None, f"{field} should be populated"
            assert 1.0 <= v <= 5.0, f"{field}={v} out of MOS range"
    else:
        # Honest state: no real model → backend unavailable, no fabricated MOS
        assert m.active_backend == "unavailable"
        for field in fields:
            assert out.quality_metrics is None or getattr(out.quality_metrics, field) is None


def test_audio_nisqa_missing_audio_is_noop(image_sample):
    from ayase.modules.audio_nisqa import AudioNISQAModule
    m = AudioNISQAModule(config={"backend": "spectral"})
    m.on_mount()
    # Image input has no audio — module must not crash, must not set fields
    out = m.process(image_sample)
    assert out.quality_metrics is None or out.quality_metrics.nisqa_mos is None
