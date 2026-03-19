"""Audio-Visual Synchronisation Detection module.

Measures the temporal offset between audio and video streams.

Algorithm:
1. Extract per-frame visual activity (mean absolute frame
   difference in luminance → "visual energy" signal).
2. Extract per-frame audio energy (RMS of audio samples
   corresponding to each frame interval).
3. Cross-correlate the two energy signals.
4. The lag at maximum correlation is the estimated A/V offset.

Result is in milliseconds — positive means audio leads video.
|offset| < 40 ms is generally imperceptible; > 100 ms is noticeable.

Requires ffmpeg for audio extraction.
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class AudioVisualSyncModule(PipelineModule):
    name = "av_sync"
    description = "Audio-video synchronisation offset detection"
    default_config = {
        "max_frames": 600,
        "warning_threshold_ms": 80.0,  # Warn if |offset| > 80 ms
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.max_frames = self.config.get("max_frames", 600)
        self.warning_threshold = self.config.get("warning_threshold_ms", 80.0)

    # ------------------------------------------------------------------
    def _extract_audio_pcm(self, video_path: Path, sr: int = 16000) -> Optional[np.ndarray]:
        """Extract raw mono PCM from video via ffmpeg."""
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()

            cmd = [
                "ffmpeg", "-y", "-i", str(video_path),
                "-vn", "-ac", "1", "-ar", str(sr),
                "-sample_fmt", "s16", tmp.name,
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            if result.returncode != 0:
                Path(tmp.name).unlink(missing_ok=True)
                return None

            import soundfile as sf

            data, _ = sf.read(tmp.name, dtype="float32")
            Path(tmp.name).unlink(missing_ok=True)

            if data.ndim > 1:
                data = data.mean(axis=1)
            return data

        except ImportError:
            logger.warning("soundfile not installed")
            return None
        except FileNotFoundError:
            logger.warning("ffmpeg not found")
            return None
        except Exception as e:
            logger.debug(f"Audio extraction failed: {e}")
            return None

    def _audio_energy_per_frame(
        self, audio: np.ndarray, sr: int, fps: float, n_frames: int
    ) -> np.ndarray:
        """Compute RMS energy of audio aligned to each video frame."""
        samples_per_frame = int(sr / fps)
        energies = []
        for i in range(n_frames):
            start = i * samples_per_frame
            end = start + samples_per_frame
            if end > len(audio):
                break
            chunk = audio[start:end]
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            energies.append(rms)
        return np.array(energies, dtype=np.float64)

    # ------------------------------------------------------------------
    def process(self, sample: Sample) -> Sample:
        if not sample.is_video:
            return sample

        # Check that the video has an audio stream
        if sample.audio_metadata is None:
            return sample

        cap = cv2.VideoCapture(str(sample.path))
        if not cap.isOpened():
            return sample

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            cap.release()
            return sample

        try:
            # --- visual energy ---
            prev_gray = None
            visual_energy = []
            idx = 0

            while idx < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)

                if prev_gray is not None:
                    diff = np.abs(gray - prev_gray)
                    visual_energy.append(float(diff.mean()))
                else:
                    visual_energy.append(0.0)

                prev_gray = gray
                idx += 1

            cap.release()

            if len(visual_energy) < 10:
                return sample

            # --- audio energy ---
            sr = 16000
            audio = self._extract_audio_pcm(sample.path, sr=sr)
            if audio is None or len(audio) < sr * 0.5:
                return sample

            n_frames = len(visual_energy)
            audio_energy = self._audio_energy_per_frame(audio, sr, fps, n_frames)

            # Align lengths
            min_len = min(len(visual_energy), len(audio_energy))
            if min_len < 10:
                return sample

            v_sig = np.array(visual_energy[:min_len])
            a_sig = audio_energy[:min_len]

            # --- cross-correlation ---
            # Normalise both signals to zero-mean, unit-variance
            v_sig = v_sig - v_sig.mean()
            a_sig = a_sig - a_sig.mean()

            v_std = v_sig.std()
            a_std = a_sig.std()

            if v_std < 1e-8 or a_std < 1e-8:
                # One signal is essentially flat — can't estimate offset
                return sample

            v_sig /= v_std
            a_sig /= a_std

            corr = np.correlate(a_sig, v_sig, mode="full")
            corr /= min_len  # normalise

            # Lag axis: negative lag means audio leads
            lags = np.arange(-(min_len - 1), min_len)
            best_lag_idx = int(np.argmax(corr))
            best_lag_frames = lags[best_lag_idx]

            offset_ms = float(best_lag_frames / fps * 1000.0)

            # Store result
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.av_sync_offset = offset_ms

            if abs(offset_ms) > self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"A/V sync offset: {offset_ms:+.1f} ms",
                        details={
                            "offset_ms": offset_ms,
                            "threshold_ms": self.warning_threshold,
                        },
                        recommendation=(
                            "Audio and video are noticeably out of sync. "
                            "Check muxing, frame rate conversion, or "
                            "audio processing pipeline."
                        ),
                    )
                )

            logger.debug(
                f"A/V sync for {sample.path.name}: {offset_ms:+.1f} ms "
                f"(lag={best_lag_frames} frames)"
            )

        except Exception as e:
            logger.error(f"A/V sync detection failed for {sample.path}: {e}")
            if cap.isOpened():
                cap.release()

        return sample


class AudioVisualSyncCompatModule(AudioVisualSyncModule):
    """Compatibility alias matching filename-based discovery."""

    name = "audio_visual_sync"
