"""MCD (Mel Cepstral Distortion) module.

Full-reference metric for evaluating TTS and voice conversion quality, computed
by ``pymcd==0.2.1``. Audio is resampled to 22,050 Hz before a WORLD spectral
envelope and 14 SPTK mel-cepstral coefficients (order 13, c0-c13) are extracted
at 5 ms intervals, using FFT size 512 and SPTK alpha 0.65. FastDTW aligns
c1-c13, while the final path distance includes
c0-c13 and uses the Kubichek constant 10·sqrt(2)/ln(10).

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
        "mcd_score": "MCD-DTW via pymcd 0.2.1 — https://github.com/chenqi008/pymcd",
    }
    deviations = {
        "mcd_score": "pymcd 0.2.1 uses FastDTW on SPTK mel-cepstral coefficients c1-c13 but includes c0-c13 in the final distance; results are not interchangeable with c0-excluding or exact-DTW MCD protocols",
    }
    description = "Mel Cepstral Distortion for TTS/VC quality (full-reference, pymcd)"
    default_config = {
        "warning_threshold": 8.0,  # dB
    }
    models = [
        {
            "id": "pymcd",
            "type": "pip_package",
            "install": "pip install pymcd==0.2.1",
            "task": "WORLD/SPTK mel-cepstrum MCD with FastDTW alignment",
            "notes": "Validated against pymcd 0.2.1; later versions may change the metric protocol",
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
                "pymcd not installed (pip install pymcd==0.2.1). "
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
