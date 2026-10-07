"""MCD (Mel Cepstral Distortion) module.

Full-reference metric for evaluating TTS and voice conversion quality,
computed by the ``pymcd`` package — 13-dimensional MFCC sequences aligned
with approximate dynamic time warping and combined with the Kubichek constant
10·sqrt(2)/ln(10) (coefficient 0 excluded). This is an adapted MCD-DTW
implementation; it is not the SPTK mel-cepstrum protocol used by many TTS
papers.

Score range: 0.0+ dB (lower = better). Values are meaningful only when the
same feature extraction, alignment, and coefficient convention is used.
Without ``pymcd`` the score is left unset.

References:
    - Kubichek (1993), "Mel-Cepstral Distance Measure for Objective
      Speech Quality Assessment"
    - https://github.com/chenqi008/pymcd (implementation used here)
"""

import logging
from pathlib import Path

import numpy as np

from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class AudioMCDModule(PipelineModule):
    name = "audio_mcd"
    provenance = "adapted"
    sources = {
        "mcd_score": "MCD-DTW via chenqi008/pymcd — https://github.com/chenqi008/pymcd",
    }
    deviations = {
        "mcd_score": "pymcd uses 13-dimensional librosa MFCCs and FastDTW; results are not interchangeable with SPTK mel-cepstrum MCD protocols",
    }
    description = "Mel Cepstral Distortion for TTS/VC quality (full-reference, pymcd)"
    default_config = {
        "warning_threshold": 8.0,  # dB
    }
    models = [
        {
            "id": "pymcd",
            "type": "pip_package",
            "install": "pip install pymcd",
            "task": "MFCC-based MCD-DTW implementation",
        },
    ]
    metric_groups = {
        "mcd_score": "audio",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.warning_threshold = self.config.get("warning_threshold", 8.0)
        self._ml_available = False
        self._backend = "unavailable"

    def setup(self) -> None:
        try:
            from pymcd.mcd import Calculate_MCD  # noqa: F401

            self._ml_available = True
            self._backend = "pymcd"
            logger.info("MCD module initialised (pymcd)")
        except ImportError:
            logger.warning(
                "pymcd not installed (pip install pymcd). "
                "MCD requires the configured pymcd backend; score left unset."
            )

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
            from pymcd.mcd import Calculate_MCD

            mcd_toolbox = Calculate_MCD(MCD_mode="dtw")
            mcd = mcd_toolbox.calculate_mcd(str(reference), str(sample.path))
            if mcd is None or not np.isfinite(mcd):
                return sample

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
