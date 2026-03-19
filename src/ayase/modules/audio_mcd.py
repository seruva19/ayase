"""MCD (Mel Cepstral Distortion) module.

Full-reference metric for evaluating TTS and voice conversion quality.
Computes the Euclidean distance between MFCC vectors of reference and
synthesised speech.

Score range: 0.0+ dB (lower = better).
  <4 dB   excellent (near-natural speech)
  4-6 dB  good
  6-8 dB  acceptable
  >8 dB   poor

Standard metric in TTS research since Kubichek (1993).

References:
    - Kubichek (1993), "Mel-Cepstral Distance Measure for Objective
      Speech Quality Assessment"
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class AudioMCDModule(PipelineModule):
    name = "audio_mcd"
    description = "Mel Cepstral Distortion for TTS/VC quality (full-reference)"
    default_config = {
        "target_sr": 16000,
        "n_mfcc": 13,
        "warning_threshold": 8.0,  # dB
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.target_sr = self.config.get("target_sr", 16000)
        self.n_mfcc = self.config.get("n_mfcc", 13)
        self.warning_threshold = self.config.get("warning_threshold", 8.0)
        self._ml_available = False

    def setup(self) -> None:
        try:
            import librosa  # noqa: F401

            self._ml_available = True
            logger.info("MCD module initialised (librosa)")
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
            ref_mfcc = self._extract_mfcc(reference)
            syn_mfcc = self._extract_mfcc(sample.path)

            if ref_mfcc is None or syn_mfcc is None:
                return sample

            # Align to shorter
            min_frames = min(ref_mfcc.shape[1], syn_mfcc.shape[1])
            if min_frames < 10:
                return sample

            ref_mfcc = ref_mfcc[:, :min_frames]
            syn_mfcc = syn_mfcc[:, :min_frames]

            # MCD: (10 * sqrt(2) / ln(10)) * mean(||ref - syn||_2)
            # Exclude c0 (energy), use c1..c_n_mfcc
            diff = ref_mfcc[1:, :] - syn_mfcc[1:, :]
            frame_dist = np.sqrt(np.sum(diff ** 2, axis=0))
            mcd = (10.0 * np.sqrt(2.0) / np.log(10.0)) * np.mean(frame_dist)

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.mcd_score = round(float(mcd), 3)

            if mcd > self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"High Mel Cepstral Distortion: {mcd:.2f} dB",
                        details={"mcd": float(mcd)},
                    )
                )

        except Exception as e:
            logger.warning(f"MCD failed for {sample.path}: {e}")

        return sample

    def _extract_mfcc(self, path: Path) -> Optional[np.ndarray]:
        try:
            import librosa

            y, sr = librosa.load(str(path), sr=self.target_sr, mono=True)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc + 1)
            return mfcc
        except Exception as e:
            logger.debug(f"MFCC extraction failed: {e}")
            return None
