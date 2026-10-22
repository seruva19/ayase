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
    def test_unavailable_without_native_audio_backend(self, synthetic_wav, monkeypatch):
        import builtins

        from ayase.modules.audio_mcd import AudioMCDModule

        original_import = builtins.__import__

        def import_without_pymcd(name, *args, **kwargs):
            if name in {"pyworld", "fastdtw"}:
                raise ImportError("simulated missing native audio backend")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", import_without_pymcd)
        mod = AudioMCDModule()
        mod.setup()
        sample = Sample(path=synthetic_wav, is_video=False, reference_path=synthetic_wav)
        result = mod.process(sample)
        assert result is sample
        assert mod._backend == "unavailable"
        assert sample.quality_metrics is None or sample.quality_metrics.mcd_score is None

    def test_pymcd_protocol_metadata(self):
        import sys

        from ayase.modules.audio_mcd import AudioMCDModule

        module_doc = sys.modules[AudioMCDModule.__module__].__doc__
        protocol = AudioMCDModule.deviations["mcd_score"]
        assert "WORLD spectral" in module_doc
        assert "14 SPTK mel-cepstral coefficients" in module_doc
        assert "pymcd 0.2.1" in protocol
        assert "c1-c13" in protocol
        assert "c0-c13" in protocol
        assert AudioMCDModule.models[0]["install"] == "managed automatically by Ayase"

    def test_self_mcd_zero_with_mocked_features(self, synthetic_wav, monkeypatch):
        import sys
        import types

        from ayase.modules.audio_mcd import AudioMCDModule, _PyMCD021

        monkeypatch.setattr(
            _PyMCD021,
            "_features",
            lambda _self, _path: np.arange(42, dtype=float).reshape(3, 14),
        )
        fake_fastdtw = types.ModuleType("fastdtw")
        fake_fastdtw.fastdtw = lambda left, right, dist: (
            0.0,
            [(index, index) for index in range(min(len(left), len(right)))],
        )
        monkeypatch.setitem(sys.modules, "fastdtw", fake_fastdtw)

        mod = AudioMCDModule()
        mod._ml_available = True
        mod._backend = "pymcd_0_2_1_port"
        sample = Sample(path=synthetic_wav, is_video=False, reference_path=synthetic_wav)
        mod.process(sample)
        assert sample.quality_metrics.mcd_score == 0.0

    def test_zero_iteration_mcep_matches_sptk_golden(self):
        from ayase.modules.audio_mcd import _pymcd_mcep

        spectrum = np.linspace(0.25, 2.0, 257, dtype=np.float64)[None, :]
        expected = np.array(
            [
                -0.6124890014713906,
                -0.7380488198682818,
                0.15105879967436936,
                -0.14527030448416361,
                0.049983657561173706,
                -0.05542661589467337,
                0.021496038515093512,
                -0.027334949919300695,
                0.011011846786718489,
                -0.015821142151043008,
                0.006443293972305832,
                -0.010236493191407478,
                0.004171622672886693,
                -0.007171775095520033,
            ]
        )
        np.testing.assert_allclose(_pymcd_mcep(spectrum)[0], expected, rtol=0.0, atol=1e-14)


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
