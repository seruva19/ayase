"""Audiobox Aesthetics — Meta Audio Aesthetics (2025).

Predicts aesthetic quality scores for audio: production quality and
enjoyment/engagement. Based on Meta's Audiobox Aesthetics model.

pip install audiobox_aesthetics

Outputs:
  audiobox_production — production quality score
  audiobox_enjoyment  — enjoyment/engagement score
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class AudioboxAestheticsModule(PipelineModule):
    name = "audiobox_aesthetics"
    description = "Meta Audiobox Aesthetics audio quality (2025)"
    default_config = {
        "sample_rate": 16000,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._model = None
        self._ml_available = False
        self._backend = None
        self.sample_rate = self.config.get("sample_rate", 16000)

    def setup(self) -> None:
        # Tier 1: audiobox_aesthetics package
        try:
            from audiobox_aesthetics import AudioBoxAesthetics
            self._model = AudioBoxAesthetics()
            self._ml_available = True
            self._backend = "audiobox"
            logger.info("Audiobox Aesthetics initialised (audiobox_aesthetics package)")
            return
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"audiobox_aesthetics init failed: {e}")

        # Tier 2: heuristic (spectral features)
        logger.info("Audiobox Aesthetics initialised (heuristic fallback)")

    def _load_audio(self, path: Path) -> Optional[np.ndarray]:
        """Load audio waveform from file."""
        try:
            import soundfile as sf
            audio, sr = sf.read(str(path))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            return audio.astype(np.float32)
        except Exception:
            pass

        # Extract from video
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            cmd = [
                "ffmpeg", "-y", "-i", str(path),
                "-vn", "-ac", "1", "-ar", str(self.sample_rate),
                "-sample_fmt", "s16", tmp.name,
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            if result.returncode != 0:
                Path(tmp.name).unlink(missing_ok=True)
                return None

            import soundfile as sf
            audio, _ = sf.read(tmp.name, dtype="float32")
            Path(tmp.name).unlink(missing_ok=True)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            return audio
        except Exception:
            pass

        return None

    def _score_audiobox(self, path: Path) -> Optional[dict]:
        """Score using audiobox_aesthetics package."""
        try:
            result = self._model.predict(str(path))
            if isinstance(result, dict):
                return {
                    "production": float(result.get("production_quality", result.get("production", 0))),
                    "enjoyment": float(result.get("enjoyment", result.get("engagement", 0))),
                }
            return None
        except Exception as e:
            logger.debug(f"Audiobox package scoring failed: {e}")
            return None

    def _score_heuristic(self, audio: np.ndarray) -> dict:
        """Heuristic: spectral features mapped to aesthetic scores.

        Production quality: based on spectral balance, dynamic range, noise floor.
        Enjoyment: based on harmonic content, rhythm regularity, spectral variety.
        """
        if len(audio) < 1024:
            return {"production": 0.5, "enjoyment": 0.5}

        # Compute spectrogram
        n_fft = 1024
        hop = 512
        n_frames = (len(audio) - n_fft) // hop + 1
        if n_frames < 1:
            return {"production": 0.5, "enjoyment": 0.5}

        frames = np.stack([
            audio[i * hop:i * hop + n_fft] * np.hanning(n_fft)
            for i in range(n_frames)
        ])
        spectrogram = np.abs(np.fft.rfft(frames, axis=1))

        # Production quality indicators
        # Dynamic range
        rms_per_frame = np.sqrt(np.mean(frames ** 2, axis=1))
        dynamic_range = float(np.log10(rms_per_frame.max() / (rms_per_frame.min() + 1e-8) + 1e-8))
        dynamic_range = min(dynamic_range / 3.0, 1.0)

        # Spectral flatness (noise-like = low quality)
        geo_mean = np.exp(np.mean(np.log(spectrogram + 1e-8), axis=1))
        arith_mean = np.mean(spectrogram, axis=1) + 1e-8
        flatness = float(np.mean(geo_mean / arith_mean))
        spectral_quality = 1.0 - flatness  # Less flat = better production

        # SNR proxy
        sorted_rms = np.sort(rms_per_frame)
        noise_floor = np.mean(sorted_rms[:max(1, len(sorted_rms) // 10)])
        signal_level = np.mean(sorted_rms[-max(1, len(sorted_rms) // 4):])
        snr_proxy = min(signal_level / (noise_floor + 1e-8) / 20.0, 1.0)

        production = float(np.clip(
            0.35 * spectral_quality + 0.35 * snr_proxy + 0.3 * dynamic_range,
            0.0, 1.0,
        ))

        # Enjoyment indicators
        # Spectral variety across time
        spectral_var = float(np.std(spectrogram, axis=0).mean())
        spectral_variety = min(spectral_var / 10.0, 1.0)

        # Rhythmic regularity: autocorrelation of energy envelope
        if len(rms_per_frame) > 10:
            env = rms_per_frame - rms_per_frame.mean()
            acorr = np.correlate(env, env, mode="full")
            acorr = acorr[len(acorr) // 2:]
            acorr /= acorr[0] + 1e-8
            # Look for periodicity peak
            if len(acorr) > 5:
                peaks = acorr[2:]
                rhythm = float(np.max(peaks)) if len(peaks) > 0 else 0.0
            else:
                rhythm = 0.0
        else:
            rhythm = 0.0

        enjoyment = float(np.clip(
            0.4 * spectral_variety + 0.3 * rhythm + 0.3 * production,
            0.0, 1.0,
        ))

        return {"production": production, "enjoyment": enjoyment}

    def process(self, sample: Sample) -> Sample:
        try:
            if self._ml_available and self._backend == "audiobox":
                scores = self._score_audiobox(sample.path)
                if scores is None:
                    return sample
            else:
                audio = self._load_audio(sample.path)
                if audio is None:
                    return sample
                scores = self._score_heuristic(audio)

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.audiobox_production = scores.get("production")
            sample.quality_metrics.audiobox_enjoyment = scores.get("enjoyment")

            logger.debug(
                f"Audiobox for {sample.path.name}: "
                f"prod={scores.get('production'):.3f} enjoy={scores.get('enjoyment'):.3f}"
            )
        except Exception as e:
            logger.error(f"Audiobox Aesthetics failed: {e}")
        return sample
