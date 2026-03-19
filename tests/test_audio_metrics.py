"""Smoke tests for audio metric modules."""

import numpy as np
import pytest
import soundfile as sf


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
    def test_identical_zero(self, synthetic_wav):
        from ayase.modules.audio_mcd import AudioMCDModule

        mod = AudioMCDModule()
        mod._ml_available = True
        mfcc = mod._extract_mfcc(synthetic_wav)
        assert mfcc is not None
        assert mfcc.shape[0] == 14  # n_mfcc + 1

    def test_self_mcd_near_zero(self, synthetic_wav):
        from ayase.modules.audio_mcd import AudioMCDModule

        mod = AudioMCDModule()
        mod._ml_available = True
        mfcc = mod._extract_mfcc(synthetic_wav)
        diff = mfcc[1:, :] - mfcc[1:, :]
        frame_dist = np.sqrt(np.sum(diff ** 2, axis=0))
        mcd = (10.0 * np.sqrt(2.0) / np.log(10.0)) * np.mean(frame_dist)
        assert mcd == 0.0


class TestAudioLPDist:
    def test_self_distance_zero(self, synthetic_wav):
        from ayase.modules.audio_lpdist import AudioLPDistModule

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
