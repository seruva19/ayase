"""PESQ (Perceptual Evaluation of Speech Quality) module.

ITU-T P.862 standard for objective speech quality measurement.
Full-reference metric that compares a degraded audio signal to an
original / reference.

Score range: -0.5 to 4.5 (higher = better).
  4.0+  excellent
  3.5+  good
  3.0+  fair
  <3.0  poor

Requires a reference audio or video with audio.  Uses the ``pesq``
Python package.
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from ayase.models import Sample, QualityMetrics, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# PESQ operates at either 8000 or 16000 Hz
_PESQ_RATES = {8000, 16000}


class AudioPESQModule(PipelineModule):
    name = "audio_pesq"
    description = "PESQ speech quality (full-reference, ITU-T P.862)"
    default_config = {
        "target_sr": 16000,  # Resample to 16 kHz (wide-band PESQ)
        "warning_threshold": 3.0,  # Warn if PESQ < 3.0
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.target_sr = self.config.get("target_sr", 16000)
        self.warning_threshold = self.config.get("warning_threshold", 3.0)
        self._ml_available = False
        self._pesq_fn = None

    def setup(self) -> None:
        try:
            from pesq import pesq as pesq_fn

            self._pesq_fn = pesq_fn
            self._ml_available = True
            logger.info("PESQ module initialised")

        except ImportError:
            logger.warning("pesq not installed. Install with: pip install pesq")
        except Exception as e:
            logger.warning(f"Failed to setup PESQ: {e}")

    # ------------------------------------------------------------------
    def _extract_audio(self, media_path: Path) -> Optional[Path]:
        """Extract audio from a video file to a temporary WAV.

        Returns path to the temp WAV, or None on failure.
        """
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()

            cmd = [
                "ffmpeg", "-y", "-i", str(media_path),
                "-vn",  # no video
                "-ac", "1",  # mono
                "-ar", str(self.target_sr),
                "-sample_fmt", "s16",
                tmp.name,
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                logger.debug(f"ffmpeg audio extraction failed: {result.stderr[:200]}")
                Path(tmp.name).unlink(missing_ok=True)
                return None

            return Path(tmp.name)

        except FileNotFoundError:
            logger.warning("ffmpeg not found — cannot extract audio")
            return None
        except Exception as e:
            logger.debug(f"Audio extraction error: {e}")
            return None

    def _load_audio(self, wav_path: Path) -> Optional[np.ndarray]:
        """Load a WAV file as a float32 NumPy array."""
        try:
            import soundfile as sf

            data, sr = sf.read(str(wav_path), dtype="float32")
            if sr != self.target_sr:
                # Resample with librosa if necessary
                import librosa

                data = librosa.resample(data, orig_sr=sr, target_sr=self.target_sr)
            # Ensure mono
            if data.ndim > 1:
                data = data.mean(axis=1)
            return data.astype(np.float32)

        except ImportError:
            logger.warning("soundfile / librosa not installed")
            return None
        except Exception as e:
            logger.debug(f"Audio load error: {e}")
            return None

    # ------------------------------------------------------------------
    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        # We need a reference to compute PESQ
        reference = getattr(sample, "reference_path", None)
        if reference is None:
            return sample
        reference = Path(reference) if not isinstance(reference, Path) else reference
        if not reference.exists():
            return sample

        try:
            # Extract / load audio from both paths
            if sample.is_video:
                dist_wav = self._extract_audio(sample.path)
                ref_wav = self._extract_audio(reference)
            else:
                # Assume audio files directly
                dist_wav = sample.path
                ref_wav = reference

            if dist_wav is None or ref_wav is None:
                return sample

            ref_audio = self._load_audio(ref_wav)
            dist_audio = self._load_audio(dist_wav)

            # Clean up temp files
            if sample.is_video:
                if dist_wav != sample.path:
                    dist_wav.unlink(missing_ok=True)
                if ref_wav != reference:
                    ref_wav.unlink(missing_ok=True)

            if ref_audio is None or dist_audio is None:
                return sample

            # Align lengths (truncate to shorter)
            min_len = min(len(ref_audio), len(dist_audio))
            if min_len < self.target_sr * 0.5:
                # Less than 0.5s of audio — too short for PESQ
                return sample

            ref_audio = ref_audio[:min_len]
            dist_audio = dist_audio[:min_len]

            # Compute PESQ
            mode = "wb" if self.target_sr == 16000 else "nb"
            pesq_score = self._pesq_fn(self.target_sr, ref_audio, dist_audio, mode)

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.pesq_score = float(pesq_score)

            if pesq_score < self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low PESQ: {pesq_score:.2f}",
                        details={"pesq": pesq_score, "threshold": self.warning_threshold},
                        recommendation=(
                            "Speech quality degraded compared to reference. "
                            "Check for noise, clipping, or codec artefacts."
                        ),
                    )
                )

            logger.debug(f"PESQ for {sample.path.name}: {pesq_score:.2f}")

        except Exception as e:
            logger.error(f"PESQ failed for {sample.path}: {e}")

        return sample
