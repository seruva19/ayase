"""CGVQM (Computer Graphics Video Quality Metric) module.

CGVQM is Intel's AI-powered metric designed for gaming and rendered
content. It detects artifacts from upscalers (DLSS, XeSS, FSR) and
frame generation that traditional metrics miss.

Range: higher = better quality.

Requires the ``cgvqm`` package from Intel. When it is unavailable the
metric is left ``None`` (no approximation is substituted).
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.base_modules import ReferenceBasedModule

logger = logging.getLogger(__name__)


class CGVQMModule(ReferenceBasedModule):
    name = "cgvqm"
    description = "CGVQM gaming/rendering quality metric (Intel, higher=better)"
    default_config = {
        "subsample": 5,
    }
    metric_groups = {
        "cgvqm": "fr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 5)
        self._ml_available = False
        self._backend = None

    def setup(self) -> None:
        try:
            import cgvqm  # noqa: F401
            self._backend = "cgvqm"
            self._ml_available = True
            logger.info("CGVQM module initialised (Intel package)")
            return
        except ImportError:
            pass

        # No real CGVQM backend available -> metric stays None.
        self._backend = "unavailable"
        self._ml_available = False
        logger.warning("CGVQM unavailable: install the Intel `cgvqm` package")

    def compute_reference_score(
        self, sample_path: Path, reference_path: Path
    ) -> Optional[float]:
        if self._backend != "cgvqm":
            return None
        ref = cv2.imread(str(reference_path))
        dist = cv2.imread(str(sample_path))
        if ref is None or dist is None:
            return None

        h = min(ref.shape[0], dist.shape[0])
        w = min(ref.shape[1], dist.shape[1])
        ref = cv2.resize(ref, (w, h))
        dist = cv2.resize(dist, (w, h))

        return self._compute_cgvqm(ref, dist)

    def _compute_cgvqm(self, ref_bgr, dist_bgr) -> Optional[float]:
        try:
            import cgvqm
            return float(cgvqm.compute(ref_bgr, dist_bgr))
        except Exception as e:
            logger.debug(f"CGVQM native failed: {e}")
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
            if sample.is_video:
                score = self._process_video(sample.path, reference)
            else:
                score = self.compute_reference_score(sample.path, reference)

            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.cgvqm = score
                logger.debug(f"CGVQM for {sample.path.name}: {score:.1f}")
        except Exception as e:
            logger.error(f"CGVQM failed: {e}")
        return sample

    def _process_video(self, path, ref_path) -> Optional[float]:
        if self._backend != "cgvqm":
            return None
        ref_cap = cv2.VideoCapture(str(ref_path))
        dist_cap = cv2.VideoCapture(str(path))
        scores = []
        idx = 0
        try:
            while True:
                r1, rf = ref_cap.read()
                r2, df = dist_cap.read()
                if not r1 or not r2:
                    break
                if idx % self.subsample == 0:
                    h = min(rf.shape[0], df.shape[0])
                    w = min(rf.shape[1], df.shape[1])
                    rf_r = cv2.resize(rf, (w, h))
                    df_r = cv2.resize(df, (w, h))
                    s = self._compute_cgvqm(rf_r, df_r)
                    if s is not None:
                        scores.append(s)
                idx += 1
        finally:
            ref_cap.release()
            dist_cap.release()
        return float(np.mean(scores)) if scores else None
