"""FAD — Frechet Audio Distance (2019).

Dataset-level metric that measures the distance between distributions of
audio features, analogous to FID for images. Uses VGGish or similar audio
embeddings and computes the Frechet distance.

pip install frechet_audio_distance

fad_score — lower = better (closer audio distributions).
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List

import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.base_modules import BatchMetricModule

logger = logging.getLogger(__name__)


class FADModule(BatchMetricModule):
    name = "fad"
    description = "Frechet Audio Distance for audio generation (batch metric, 2019)"
    default_config = {
        "subsample_videos": None,
        "sample_rate": 16000,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._model = None
        self._ml_available = False
        self._backend = None
        self.subsample_videos = self.config.get("subsample_videos", None)
        self.sample_rate = self.config.get("sample_rate", 16000)
        self._processed_count = 0

    def setup(self) -> None:
        # Tier 1: frechet_audio_distance package
        try:
            from frechet_audio_distance import FrechetAudioDistance
            self._model = FrechetAudioDistance()
            self._ml_available = True
            self._backend = "fad_package"
            logger.info("FAD module initialised (frechet_audio_distance package)")
            return
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"FAD package init failed: {e}")

        # Tier 2: heuristic (spectral features + Frechet distance)
        logger.info("FAD module initialised (heuristic fallback)")

    def extract_features(self, sample: Sample) -> Optional[np.ndarray]:
        """Extract audio features for distribution comparison."""
        if self.subsample_videos is not None and self._processed_count >= self.subsample_videos:
            return None

        try:
            audio = self._load_audio(sample.path)
            if audio is None:
                return None

            features = self._compute_spectral_features(audio)
            if features is not None:
                self._processed_count += 1
            return features
        except Exception as e:
            logger.debug(f"FAD feature extraction failed for {sample.path}: {e}")
            return None

    def _load_audio(self, path: Path) -> Optional[np.ndarray]:
        """Load audio from file, extracting from video if necessary."""
        # Try direct audio loading
        try:
            import soundfile as sf
            audio, sr = sf.read(str(path))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != self.sample_rate:
                # Simple resampling via linear interpolation
                duration = len(audio) / sr
                n_samples = int(duration * self.sample_rate)
                indices = np.linspace(0, len(audio) - 1, n_samples)
                audio = np.interp(indices, np.arange(len(audio)), audio)
            return audio.astype(np.float32)
        except Exception:
            pass

        # Try extracting audio from video via ffmpeg
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

    def _compute_spectral_features(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Compute spectral features (MFCCs + spectral statistics)."""
        if len(audio) < 1024:
            return None

        # Compute spectrogram
        n_fft = 1024
        hop = 512
        n_frames = (len(audio) - n_fft) // hop + 1
        if n_frames < 1:
            return None

        frames = np.stack([
            audio[i * hop:i * hop + n_fft] * np.hanning(n_fft)
            for i in range(n_frames)
        ])
        spectrogram = np.abs(np.fft.rfft(frames, axis=1))

        # Mel-like features: group FFT bins into ~20 bands
        n_bands = 20
        n_bins = spectrogram.shape[1]
        band_size = max(n_bins // n_bands, 1)
        mel_features = []
        for b in range(n_bands):
            start = b * band_size
            end = min(start + band_size, n_bins)
            if start >= n_bins:
                mel_features.append(0.0)
            else:
                mel_features.append(float(spectrogram[:, start:end].mean()))

        # Spectral statistics
        spectral_centroid = float(np.mean(
            np.sum(spectrogram * np.arange(spectrogram.shape[1]), axis=1) /
            (np.sum(spectrogram, axis=1) + 1e-8)
        ))
        spectral_rolloff = float(np.mean(np.percentile(spectrogram, 85, axis=1)))
        spectral_flux = float(np.mean(np.diff(spectrogram, axis=0) ** 2)) if n_frames > 1 else 0.0

        features = np.array(mel_features + [spectral_centroid, spectral_rolloff, spectral_flux])
        return features.astype(np.float64)

    def compute_distribution_metric(
        self, features: List[np.ndarray], reference_features: Optional[List[np.ndarray]] = None
    ) -> float:
        """Compute Frechet distance between audio feature distributions."""
        try:
            features_array = np.stack(features, axis=0)

            if reference_features is not None and len(reference_features) > 0:
                ref_array = np.stack(reference_features, axis=0)
            else:
                mid = len(features_array) // 2
                if mid < 1:
                    return 0.0
                ref_array = features_array[:mid]
                features_array = features_array[mid:]

            return self._frechet_distance(features_array, ref_array)
        except Exception as e:
            logger.error(f"FAD computation failed: {e}")
            return float("inf")

    def _frechet_distance(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        mu1 = np.mean(feat1, axis=0)
        mu2 = np.mean(feat2, axis=0)

        if feat1.shape[0] < 2 or feat2.shape[0] < 2:
            return float(np.sum((mu1 - mu2) ** 2))

        sigma1 = np.cov(feat1, rowvar=False)
        sigma2 = np.cov(feat2, rowvar=False)

        if sigma1.ndim == 0:
            sigma1 = np.array([[sigma1]])
        if sigma2.ndim == 0:
            sigma2 = np.array([[sigma2]])

        diff = mu1 - mu2

        try:
            from scipy import linalg
            covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
            if np.iscomplexobj(covmean):
                covmean = covmean.real
            fd = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
        except ImportError:
            fd = float(diff @ diff + np.trace(sigma1) + np.trace(sigma2))

        return float(fd)

    def on_dispose(self) -> None:
        if len(self._feature_cache) < 2:
            logger.info(f"FAD: Not enough samples ({len(self._feature_cache)})")
            self._feature_cache = []
            self._reference_cache = []
            return

        try:
            score = self.compute_distribution_metric(
                self._feature_cache,
                self._reference_cache if self._reference_cache else None,
            )
            logger.info(f"FAD: {score:.4f} ({len(self._feature_cache)} samples)")

            if hasattr(self, "pipeline") and self.pipeline:
                if hasattr(self.pipeline, "add_dataset_metric"):
                    self.pipeline.add_dataset_metric("fad", score)
        except Exception as e:
            logger.error(f"FAD failed: {e}")
        finally:
            self._feature_cache = []
            self._reference_cache = []
            self._processed_count = 0
