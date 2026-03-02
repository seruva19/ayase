"""Butteraugli module.

Butteraugli is Google's perceptual image distance metric, designed for
evaluating compression quality. It models the human visual system and
is used as the core metric in JPEG XL encoding decisions.

Range: 0+ (lower = better, 0 = identical).
  < 0.5: imperceptible difference
  0.5-1.0: barely noticeable
  1.0-2.0: noticeable
  > 2.0: clearly visible distortion

This is a full-reference metric.

Requires one of:
  - ``jxlpy`` (JPEG XL Python bindings with butteraugli)
  - ``butteraugli`` Python package
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics, ValidationIssue, ValidationSeverity
from ayase.base_modules import ReferenceBasedModule

logger = logging.getLogger(__name__)


class ButteraugliModule(ReferenceBasedModule):
    name = "butteraugli"
    description = "Butteraugli perceptual distance (Google/JPEG XL, lower=better)"
    default_config = {
        "subsample": 5,
        "warning_threshold": 2.0,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 5)
        self.warning_threshold = self.config.get("warning_threshold", 2.0)
        self._ml_available = False
        self._backend = None  # "jxlpy" or "butteraugli"

    def setup(self) -> None:
        # Try jxlpy first (JPEG XL bindings with butteraugli)
        try:
            import jxlpy

            self._backend = "jxlpy"
            self._ml_available = True
            logger.info("Butteraugli module initialised (jxlpy backend)")
            return
        except ImportError:
            pass

        # Try standalone butteraugli package
        try:
            import butteraugli as ba

            self._backend = "butteraugli"
            self._ml_available = True
            logger.info("Butteraugli module initialised (butteraugli backend)")
            return
        except ImportError:
            pass

        # Fallback: OpenCV-based approximation using Laplacian + edge comparison
        self._backend = "approx"
        self._ml_available = True
        logger.info("Butteraugli module initialised (OpenCV approximation fallback)")

    def compute_reference_score(
        self, sample_path: Path, reference_path: Path
    ) -> Optional[float]:
        try:
            ref_img = cv2.imread(str(reference_path))
            dist_img = cv2.imread(str(sample_path))
            if ref_img is None or dist_img is None:
                return None

            # Align sizes
            h = min(ref_img.shape[0], dist_img.shape[0])
            w = min(ref_img.shape[1], dist_img.shape[1])
            ref_img = cv2.resize(ref_img, (w, h))
            dist_img = cv2.resize(dist_img, (w, h))

            if self._backend == "jxlpy":
                return self._compute_jxlpy(ref_img, dist_img)
            elif self._backend == "butteraugli":
                return self._compute_butteraugli(ref_img, dist_img)
            else:
                return self._compute_approx(ref_img, dist_img)
        except Exception as e:
            logger.debug(f"Butteraugli scoring failed: {e}")
            return None

    def _compute_jxlpy(self, ref_bgr: np.ndarray, dist_bgr: np.ndarray) -> float:
        import jxlpy

        ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
        dist_rgb = cv2.cvtColor(dist_bgr, cv2.COLOR_BGR2RGB)
        return float(jxlpy.butteraugli_distance(ref_rgb, dist_rgb))

    def _compute_butteraugli(self, ref_bgr: np.ndarray, dist_bgr: np.ndarray) -> float:
        import butteraugli as ba

        ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
        dist_rgb = cv2.cvtColor(dist_bgr, cv2.COLOR_BGR2RGB)
        return float(ba.compute(ref_rgb, dist_rgb))

    def _compute_approx(self, ref_bgr: np.ndarray, dist_bgr: np.ndarray) -> float:
        """Approximate butteraugli using frequency-domain perceptual difference.

        This is a rough proxy, not the actual butteraugli algorithm.
        Uses multi-scale edge + colour difference analysis.
        """
        ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        dist_lab = cv2.cvtColor(dist_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Per-channel difference
        diff = np.abs(ref_lab - dist_lab)

        # Weight: L channel matters more than a/b
        weighted = diff[:, :, 0] * 0.6 + diff[:, :, 1] * 0.2 + diff[:, :, 2] * 0.2

        # Edge-weighted (perceptually important near edges)
        ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(ref_gray, 50, 150).astype(np.float32) / 255.0
        edge_weight = 1.0 + edges * 2.0  # 3x weight at edges
        weighted = weighted * edge_weight

        # Map to butteraugli-like scale (rough calibration)
        score = float(np.percentile(weighted, 99)) / 30.0
        return max(0.0, score)

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

            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.butteraugli = score

            if score > self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"High Butteraugli distance: {score:.2f}",
                        details={"butteraugli": score, "threshold": self.warning_threshold},
                        recommendation="Visible perceptual differences from reference.",
                    )
                )
            logger.debug(f"Butteraugli for {sample.path.name}: {score:.2f}")
        except Exception as e:
            logger.error(f"Butteraugli failed for {sample.path}: {e}")
        return sample

    def _process_video(self, path: Path, ref_path: Path) -> Optional[float]:
        ref_cap = cv2.VideoCapture(str(ref_path))
        dist_cap = cv2.VideoCapture(str(path))
        scores = []
        idx = 0

        while True:
            r1, ref_f = ref_cap.read()
            r2, dist_f = dist_cap.read()
            if not r1 or not r2:
                break
            if idx % self.subsample == 0:
                h = min(ref_f.shape[0], dist_f.shape[0])
                w = min(ref_f.shape[1], dist_f.shape[1])
                ref_r = cv2.resize(ref_f, (w, h))
                dist_r = cv2.resize(dist_f, (w, h))
                try:
                    if self._backend == "jxlpy":
                        s = self._compute_jxlpy(ref_r, dist_r)
                    elif self._backend == "butteraugli":
                        s = self._compute_butteraugli(ref_r, dist_r)
                    else:
                        s = self._compute_approx(ref_r, dist_r)
                    scores.append(s)
                except Exception:
                    pass
            idx += 1

        ref_cap.release()
        dist_cap.release()
        return float(np.mean(scores)) if scores else None
