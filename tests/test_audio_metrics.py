"""Smoke tests for audio metric modules."""

import numpy as np
import pytest
import soundfile as sf

from ayase.models import Sample


@pytest.fixture
def synthetic_wav(tmp_path):
    """440 Hz sine wave, 1 second, 16 kHz."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    path = tmp_path / "ref.wav"
    sf.write(str(path), audio, sr)
    return path


@pytest.fixture
def degraded_wav(tmp_path):
    """440 Hz sine + noise."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.05 * np.random.randn(sr).astype(np.float32)
    path = tmp_path / "deg.wav"
    sf.write(str(path), audio, sr)
    return path


class TestAudioSISDR:
    def test_identical_signals(self, synthetic_wav):
        from ayase.modules.audio_si_sdr import AudioSISDRModule

        mod = AudioSISDRModule()
        ref = mod._load_audio(synthetic_wav)
        si_sdr = mod._compute_si_sdr(ref, ref)
        assert si_sdr > 50  # identical → very high

    def test_noisy_signal_positive(self, synthetic_wav, degraded_wav):
        from ayase.modules.audio_si_sdr import AudioSISDRModule

        mod = AudioSISDRModule()
        ref = mod._load_audio(synthetic_wav)
        deg = mod._load_audio(degraded_wav)
        si_sdr = mod._compute_si_sdr(ref, deg)
        assert si_sdr > 0  # signal still dominates noise


class TestAudioMCD:
    def test_unavailable_without_pymcd(self, synthetic_wav):
        from ayase.modules.audio_mcd import AudioMCDModule

        mod = AudioMCDModule()
        # pymcd is not installed in the test env — score must stay unset.
        sample = Sample(path=synthetic_wav, is_video=False, reference_path=synthetic_wav)
        result = mod.process(sample)
        assert result is sample
        assert sample.quality_metrics is None or sample.quality_metrics.mcd_score is None

    def test_self_mcd_zero_with_mocked_pymcd(self, synthetic_wav, monkeypatch):
        import sys
        import types

        from ayase.modules.audio_mcd import AudioMCDModule

        class _Calc:
            def __init__(self, MCD_mode="dtw"):
                self.mode = MCD_mode

            def calculate_mcd(self, ref, deg):
                return 0.0  # identical files

        fake_mod = types.ModuleType("pymcd.mcd")
        fake_mod.Calculate_MCD = _Calc
        fake_pkg = types.ModuleType("pymcd")
        fake_pkg.mcd = fake_mod
        monkeypatch.setitem(sys.modules, "pymcd", fake_pkg)
        monkeypatch.setitem(sys.modules, "pymcd.mcd", fake_mod)

        mod = AudioMCDModule()
        mod.setup()
        sample = Sample(path=synthetic_wav, is_video=False, reference_path=synthetic_wav)
        mod.process(sample)
        assert sample.quality_metrics.mcd_score == 0.0


class TestAudioLPDist:
    def test_self_distance_zero(self, synthetic_wav):
        from ayase.modules.audio_logmel_dist import AudioLPDistModule

        mod = AudioLPDistModule()
        mod._ml_available = True
        mel = mod._extract_log_mel(synthetic_wav)
        assert mel is not None
        dist = float(np.sqrt(np.mean((mel - mel) ** 2)))
        assert dist == 0.0


class TestAudioESTOI:
    def test_import(self):
        try:
            from ayase.modules.audio_estoi import AudioESTOIModule
            mod = AudioESTOIModule()
            assert mod.name == "audio_estoi"
        except ImportError:
            pytest.skip("pystoi not installed")


class TestAudioUTMOS:
    def test_import(self):
        from ayase.modules.audio_utmos import AudioUTMOSModule
        mod = AudioUTMOSModule()
        assert mod.name == "audio_utmos"
