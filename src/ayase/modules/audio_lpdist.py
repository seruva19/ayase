"""LPDist (Log-Power Spectral Distance) module.

Full-reference metric comparing mel-spectrogram power between reference
and degraded audio.  Captures spectral envelope differences — useful for
TTS, voice conversion, and audio coding quality assessment.

Score range: 0.0+ (lower = better, 0 = identical spectra).
  <1.0   excellent
  1-2    good
  2-4    fair
  >4     poor

References:
    - Gray & Markel (1976), "Distance Measures for Speech Processing"
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class AudioLPDistModule(PipelineModule):
    name = "audio_lpdist"
    description = "Log-Power Spectral Distance (full-reference audio)"
    default_config = {
        "target_sr": 16000,
        "n_mels": 80,
        "warning_threshold": 4.0,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.target_sr = self.config.get("target_sr", 16000)
        self.n_mels = self.config.get("n_mels", 80)
        self.warning_threshold = self.config.get("warning_threshold", 4.0)
        self._ml_available = False

    def setup(self) -> None:
        try:
            import librosa  # noqa: F401

            self._ml_available = True
            logger.info("LPDist module initialised (librosa)")
        except ImportError:
            logger.warning("librosa not installed. Install with: pip install librosa")

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        reference = getattr(sample, "reference_path", None)
        if reference is None:
            return sample
        reference = Path(reference) if not isinstance(reference, Path) else reference
        if not reference.exists():
            return sample

        try:
            ref_mel = self._extract_log_mel(reference)
            deg_mel = self._extract_log_mel(sample.path)

            if ref_mel is None or deg_mel is None:
                return sample

            min_frames = min(ref_mel.shape[1], deg_mel.shape[1])
            if min_frames < 5:
                return sample

            ref_mel = ref_mel[:, :min_frames]
            deg_mel = deg_mel[:, :min_frames]

            # Mean squared difference in log-power domain
            lpdist = float(np.sqrt(np.mean((ref_mel - deg_mel) ** 2)))

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.lpdist_score = round(lpdist, 4)

            if lpdist > self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"High spectral distance (LPDist): {lpdist:.3f}",
                        details={"lpdist": lpdist},
                    )
                )

        except Exception as e:
            logger.warning(f"LPDist failed for {sample.path}: {e}")

        return sample

    def _extract_log_mel(self, path: Path) -> Optional[np.ndarray]:
        try:
            import librosa

            y, sr = librosa.load(str(path), sr=self.target_sr, mono=True)
            mel = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=self.n_mels
            )
            log_mel = librosa.power_to_db(mel, ref=np.max)
            return log_mel
        except Exception as e:
            logger.debug(f"Mel extraction failed: {e}")
            return None
