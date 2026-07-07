"""STRRED (Spatio-Temporal Reduced Reference Entropic Differencing) module.

STRRED (Soundararajan & Bovik) is a reduced-reference video quality metric
based on wavelet-domain natural-scene-statistics entropic differences in the
spatial and temporal domains.

Range: lower = better quality (0 = identical).

Uses scikit-video's ``skvideo.measure.strred`` implementation. If scikit-video
is not installed the metric is reported as unavailable (a previous revision
fell back to a mean-absolute-difference "approximation" that is not STRRED;
that proxy has been removed).
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.base_modules import ReferenceBasedModule

logger = logging.getLogger(__name__)


class STRREDModule(ReferenceBasedModule):
    name = "strred"
    description = "STRRED reduced-reference temporal quality (ITU, lower=better)"
    default_config = {
        "subsample": 3,
    }
    metric_groups = {
        "strred": "fr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 3)
        self._ml_available = False
        self._backend = None

    def setup(self) -> None:
        try:
            import skvideo.measure  # noqa: F401

            self._backend = "skvideo"
            self._ml_available = True
            logger.info("STRRED module initialised (scikit-video)")
        except ImportError:
            self._backend = "unavailable"
            logger.info(
                "STRRED: scikit-video not installed; metric unavailable. "
                "Install with: pip install scikit-video"
            )

    def compute_reference_score(
        self, sample_path: Path, reference_path: Path
    ) -> Optional[float]:
        # STRRED is a video (temporal) metric; it is not defined for still images.
        if sample_path.suffix.lower() not in ('.mp4', '.avi', '.mkv', '.mov', '.webm'):
            return None
        return self._strred_skvideo(sample_path, reference_path)

    def _strred_skvideo(self, sample_path, reference_path) -> Optional[float]:
        try:
            import skvideo.io
            import skvideo.measure

            ref_vid = skvideo.io.vread(str(reference_path), as_grey=True)
            dist_vid = skvideo.io.vread(str(sample_path), as_grey=True)

            n = min(len(ref_vid), len(dist_vid))
            if n < 2:
                return None
            ref_vid = ref_vid[:n]
            dist_vid = dist_vid[:n]

            result = skvideo.measure.strred(ref_vid, dist_vid)
            # skvideo returns (per-frame array, strred, strredsn); prefer the
            # aggregate STRRED scalar when available.
            if isinstance(result, (tuple, list)) and len(result) >= 2:
                return float(result[1])
            return float(np.mean(result))
        except Exception as e:
            logger.debug(f"STRRED skvideo failed: {e}")
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

        try:
            score = self.compute_reference_score(sample.path, reference)
            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.strred = score
                logger.debug(f"STRRED for {sample.path.name}: {score:.4f}")
        except Exception as e:
            logger.error(f"STRRED failed: {e}")
        return sample
