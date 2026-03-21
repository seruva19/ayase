"""ESTOI (Extended Short-Time Objective Intelligibility) module.

Full-reference speech intelligibility metric. Measures how well speech
can be understood after processing/degradation compared to the original.

Improvement over classic STOI: handles more types of distortion
(modulated noise, non-linear processing).

Score range: 0.0 to 1.0 (higher = more intelligible).
  0.9+  excellent intelligibility
  0.7+  good
  0.5+  fair
  <0.5  poor

Requires a reference audio file.  Uses the ``pystoi`` package.

References:
    - Jensen & Taal (2016), "An Algorithm for Predicting the
      Intelligibility of Speech Masked by Modulated Noise Maskers"
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class AudioESTOIModule(PipelineModule):
    name = "audio_estoi"
    description = "ESTOI speech intelligibility (full-reference)"
    default_config = {
        "target_sr": 10000,  # ESTOI standard sample rate
        "warning_threshold": 0.5,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.target_sr = self.config.get("target_sr", 10000)
        self.warning_threshold = self.config.get("warning_threshold", 0.5)
        self._ml_available = False
        self._stoi_fn = None

    def setup(self) -> None:
        try:
            from pystoi import stoi

            self._stoi_fn = stoi
            self._ml_available = True
            logger.info("ESTOI module initialised (pystoi)")
        except ImportError:
            logger.warning("pystoi not installed. Install with: pip install pystoi")
        except Exception as e:
            logger.warning(f"Failed to setup ESTOI: {e}")

    def _extract_audio(self, video_path: Path) -> Optional[Path]:
        """Extract audio from a video file to a temporary WAV."""
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "1",
                 "-ar", str(self.target_sr), tmp.name],
                capture_output=True, timeout=30,
            )
            if result.returncode != 0:
                Path(tmp.name).unlink(missing_ok=True)
                return None
            return Path(tmp.name)
        except Exception:
            Path(tmp.name).unlink(missing_ok=True)
            return None

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        reference = getattr(sample, "reference_path", None)
        if reference is None:
            return sample
        reference = Path(reference) if not isinstance(reference, Path) else reference
        if not reference.exists():
            return sample

        ref_tmp = None
        deg_tmp = None
        try:
            if sample.is_video:
                deg_tmp = self._extract_audio(sample.path)
                ref_tmp = self._extract_audio(reference)
                if deg_tmp is None or ref_tmp is None:
                    return sample
                ref_audio = self._load_audio(ref_tmp)
                deg_audio = self._load_audio(deg_tmp)
            else:
                ref_audio = self._load_audio(reference)
                deg_audio = self._load_audio(sample.path)

            if ref_audio is None or deg_audio is None:
                return sample

            # Align lengths
            min_len = min(len(ref_audio), len(deg_audio))
            if min_len < self.target_sr * 0.5:
                return sample

            ref_audio = ref_audio[:min_len]
            deg_audio = deg_audio[:min_len]

            # Extended STOI
            score = self._stoi_fn(
                ref_audio, deg_audio, self.target_sr, extended=True
            )

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.estoi_score = round(float(score), 4)

            if score < self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low speech intelligibility (ESTOI): {score:.3f}",
                        details={"estoi": float(score)},
                    )
                )

        except Exception as e:
            logger.warning(f"ESTOI failed for {sample.path}: {e}")
        finally:
            if ref_tmp is not None:
                ref_tmp.unlink(missing_ok=True)
            if deg_tmp is not None:
                deg_tmp.unlink(missing_ok=True)

        return sample

    def _load_audio(self, path: Path) -> Optional[np.ndarray]:
        try:
            import soundfile as sf

            data, sr = sf.read(str(path), dtype="float32")
            if sr != self.target_sr:
                import librosa

                data = librosa.resample(data, orig_sr=sr, target_sr=self.target_sr)
            if data.ndim > 1:
                data = data.mean(axis=1)
            return data.astype(np.float32)
        except ImportError:
            logger.warning("soundfile/librosa not installed")
            return None
        except Exception as e:
            logger.debug(f"Audio load error: {e}")
            return None
