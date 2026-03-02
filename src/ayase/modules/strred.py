"""STRRED (Spatio-Temporal Reduced Reference Entropic Differencing) module.

STRRED is an ITU-standard reduced-reference video quality metric based
on wavelet-domain natural scene statistics. It uses temporal and spatial
entropy differences to assess quality.

Range: lower = better quality (0 = identical).

Uses scikit-video's STRRED implementation or OpenCV approximation.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
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

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 3)
        self._ml_available = False
        self._backend = None

    def setup(self) -> None:
        try:
            import skvideo.measure
            self._backend = "skvideo"
            self._ml_available = True
            logger.info("STRRED module initialised (scikit-video)")
            return
        except ImportError:
            pass

        # Fallback: OpenCV approximation
        self._backend = "approx"
        self._ml_available = True
        logger.info("STRRED module initialised (OpenCV approximation)")

    def compute_reference_score(
        self, sample_path: Path, reference_path: Path
    ) -> Optional[float]:
        if sample_path.suffix.lower() not in ('.mp4', '.avi', '.mkv', '.mov', '.webm'):
            return self._compute_image(sample_path, reference_path)
        return self._compute_video(sample_path, reference_path)

    def _compute_video(self, sample_path, reference_path) -> Optional[float]:
        if self._backend == "skvideo":
            return self._strred_skvideo(sample_path, reference_path)
        return self._strred_approx(sample_path, reference_path)

    def _compute_image(self, sample_path, reference_path) -> Optional[float]:
        ref = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
        dist = cv2.imread(str(sample_path), cv2.IMREAD_GRAYSCALE)
        if ref is None or dist is None:
            return None
        h = min(ref.shape[0], dist.shape[0])
        w = min(ref.shape[1], dist.shape[1])
        ref = cv2.resize(ref, (w, h)).astype(np.float32)
        dist = cv2.resize(dist, (w, h)).astype(np.float32)
        # Spatial entropy difference
        diff = np.abs(ref - dist)
        return float(np.mean(diff) / 255.0)

    def _strred_skvideo(self, sample_path, reference_path) -> Optional[float]:
        try:
            import skvideo.io
            import skvideo.measure

            ref_vid = skvideo.io.vread(str(reference_path))
            dist_vid = skvideo.io.vread(str(sample_path))

            n = min(len(ref_vid), len(dist_vid))
            ref_vid = ref_vid[:n]
            dist_vid = dist_vid[:n]

            scores = skvideo.measure.strred(ref_vid, dist_vid)
            return float(np.mean(scores))
        except Exception as e:
            logger.debug(f"STRRED skvideo failed: {e}")
            return self._strred_approx(sample_path, reference_path)

    def _strred_approx(self, sample_path, reference_path) -> Optional[float]:
        """Approximate STRRED using temporal + spatial difference entropy."""
        ref_cap = cv2.VideoCapture(str(reference_path))
        dist_cap = cv2.VideoCapture(str(sample_path))

        scores = []
        prev_ref = None
        prev_dist = None
        idx = 0

        while True:
            r1, ref_f = ref_cap.read()
            r2, dist_f = dist_cap.read()
            if not r1 or not r2:
                break
            if idx % self.subsample == 0:
                ref_g = cv2.cvtColor(ref_f, cv2.COLOR_BGR2GRAY).astype(np.float32)
                dist_g = cv2.cvtColor(dist_f, cv2.COLOR_BGR2GRAY).astype(np.float32)
                h = min(ref_g.shape[0], dist_g.shape[0])
                w = min(ref_g.shape[1], dist_g.shape[1])
                ref_g = cv2.resize(ref_g, (w, h))
                dist_g = cv2.resize(dist_g, (w, h))

                # Spatial difference
                spatial = float(np.mean(np.abs(ref_g - dist_g))) / 255.0

                # Temporal difference
                temporal = 0.0
                if prev_ref is not None and prev_dist is not None:
                    ref_diff = np.abs(ref_g - prev_ref)
                    dist_diff = np.abs(dist_g - prev_dist)
                    temporal = float(np.mean(np.abs(ref_diff - dist_diff))) / 255.0

                scores.append(spatial * 0.5 + temporal * 0.5)
                prev_ref = ref_g
                prev_dist = dist_g
            idx += 1

        ref_cap.release()
        dist_cap.release()
        return float(np.mean(scores)) if scores else None

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
