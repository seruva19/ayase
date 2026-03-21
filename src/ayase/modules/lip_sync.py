"""Lip Sync Error — LSE-D / LSE-C (SyncNet / Wav2Lip, 2020).

Measures audio-visual lip synchronisation quality:
  LSE-D (Lip Sync Error - Distance): lower = better sync
  LSE-C (Lip Sync Error - Confidence): higher = better sync

Uses SyncNet-based scoring or falls back to audio-video temporal
correlation heuristic.

pip install syncnet (or wav2lip)
"""

import logging
import subprocess
import tempfile
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class LipSyncModule(PipelineModule):
    name = "lip_sync"
    description = "LSE-D/LSE-C lip sync error (SyncNet/Wav2Lip, 2020)"
    default_config = {
        "subsample": 16,
        "sample_rate": 16000,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._model = None
        self._ml_available = False
        self._backend = None
        self.subsample = self.config.get("subsample", 16)
        self.sample_rate = self.config.get("sample_rate", 16000)

    def setup(self) -> None:
        # Tier 1: syncnet package
        try:
            import syncnet
            self._model = syncnet
            self._ml_available = True
            self._backend = "syncnet"
            logger.info("Lip Sync module initialised (syncnet package)")
            return
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"syncnet init failed: {e}")

        # Tier 2: heuristic (audio-video temporal correlation)
        logger.info("Lip Sync module initialised (heuristic fallback)")

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video:
            return sample

        try:
            if self._ml_available and self._backend == "syncnet":
                scores = self._score_syncnet(sample.path)
            else:
                scores = self._score_heuristic(sample.path)

            if scores is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.lse_d = scores.get("lse_d")
            sample.quality_metrics.lse_c = scores.get("lse_c")
            logger.debug(
                f"Lip Sync for {sample.path.name}: "
                f"LSE-D={scores.get('lse_d', 0.0):.4f} LSE-C={scores.get('lse_c', 0.0):.4f}"
            )
        except Exception as e:
            logger.error(f"Lip Sync failed: {e}")
        return sample

    def _score_syncnet(self, path: Path) -> Optional[dict]:
        try:
            result = self._model.evaluate(str(path))
            return {
                "lse_d": float(result.get("lse_d", result.get("distance", 0))),
                "lse_c": float(result.get("lse_c", result.get("confidence", 0))),
            }
        except Exception as e:
            logger.debug(f"SyncNet scoring failed: {e}")
            return self._score_heuristic(path)

    def _score_heuristic(self, path: Path) -> Optional[dict]:
        """Heuristic: correlate mouth-region visual activity with audio energy.

        LSE-D proxy: inverse correlation between mouth motion and audio energy.
        LSE-C proxy: peak cross-correlation value.
        """
        # Extract visual mouth-region activity
        mouth_energy = self._extract_mouth_energy(path)
        if mouth_energy is None or len(mouth_energy) < 5:
            return None

        # Extract audio energy
        audio_energy = self._extract_audio_energy(path, len(mouth_energy))
        if audio_energy is None or len(audio_energy) < 5:
            return None

        # Align lengths
        min_len = min(len(mouth_energy), len(audio_energy))
        v_sig = mouth_energy[:min_len].astype(np.float64)
        a_sig = audio_energy[:min_len].astype(np.float64)

        # Normalise
        v_sig -= v_sig.mean()
        a_sig -= a_sig.mean()
        v_std = v_sig.std()
        a_std = a_sig.std()

        if v_std < 1e-8 or a_std < 1e-8:
            return {"lse_d": 10.0, "lse_c": 0.0}

        v_sig /= v_std
        a_sig /= a_std

        # Cross-correlation
        corr = np.correlate(a_sig, v_sig, mode="full")
        corr /= min_len

        peak_corr = float(np.max(corr))
        best_lag = int(np.argmax(corr)) - (min_len - 1)

        # LSE-D: distance proxy (lower = better sync)
        # Map: high correlation at small lag → low distance
        lse_d = max(0.0, 1.0 - peak_corr) * 10.0 + abs(best_lag) * 0.5

        # LSE-C: confidence proxy (higher = better sync)
        lse_c = float(np.clip(peak_corr, 0.0, 1.0)) * 10.0

        return {"lse_d": lse_d, "lse_c": lse_c}

    def _extract_mouth_energy(self, path: Path) -> Optional[np.ndarray]:
        """Extract per-frame mouth region motion energy."""
        cap = cv2.VideoCapture(str(path))
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total < 2:
                return None

            n_sample = min(self.subsample, total)
            indices = np.linspace(0, total - 1, n_sample, dtype=int)

            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

            prev_mouth = None
            energies = []

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                if len(faces) > 0:
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    # Lower half of face = mouth region
                    mouth_y = y + int(h * 0.6)
                    mouth_roi = gray[mouth_y:y + h, x:x + w]
                    if mouth_roi.size > 0:
                        mouth_resized = cv2.resize(mouth_roi, (32, 16)).astype(np.float64)
                    else:
                        mouth_resized = np.zeros((16, 32), dtype=np.float64)
                else:
                    # Fallback: use bottom-center region
                    fh, fw = gray.shape
                    mouth_resized = cv2.resize(
                        gray[int(fh * 0.6):fh, int(fw * 0.25):int(fw * 0.75)],
                        (32, 16)
                    ).astype(np.float64)

                if prev_mouth is not None:
                    energy = float(np.mean(np.abs(mouth_resized - prev_mouth)))
                    energies.append(energy)
                else:
                    energies.append(0.0)

                prev_mouth = mouth_resized

            return np.array(energies) if energies else None
        finally:
            cap.release()

    def _extract_audio_energy(self, path: Path, n_frames: int) -> Optional[np.ndarray]:
        """Extract per-frame audio energy."""
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
            audio, sr = sf.read(tmp.name, dtype="float32")
            Path(tmp.name).unlink(missing_ok=True)

            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            # Compute per-frame energy
            samples_per_frame = len(audio) // n_frames
            if samples_per_frame < 1:
                return None

            energies = []
            for i in range(n_frames):
                start = i * samples_per_frame
                end = start + samples_per_frame
                if end > len(audio):
                    break
                chunk = audio[start:end]
                rms = float(np.sqrt(np.mean(chunk ** 2)))
                energies.append(rms)

            return np.array(energies)
        except Exception as e:
            logger.debug(f"Audio energy extraction failed: {e}")
            return None
