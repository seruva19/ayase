"""SSIMULACRA 2 module.

SSIMULACRA 2 is Google's perceptual image quality metric developed for
the JPEG XL codec. It is based on a multi-scale structural similarity
computation with psychovisual error modeling.

Range: roughly -inf to 100 (lower = better perceptual quality).
  Score < 0: distortions not likely visible
  0-30: low quality
  30-70: medium quality
  70-100: high quality (significant distortion)

This is a full-reference metric.

Requires the ``ssimulacra2`` Python package:
  pip install ssimulacra2
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics, ValidationIssue, ValidationSeverity
from ayase.base_modules import ReferenceBasedModule

logger = logging.getLogger(__name__)


class SSIMULACRA2Module(ReferenceBasedModule):
    name = "ssimulacra2"
    description = "SSIMULACRA 2 perceptual distance (JPEG XL standard, lower=better)"
    default_config = {
        "subsample": 5,
        "warning_threshold": 50.0,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 5)
        self.warning_threshold = self.config.get("warning_threshold", 50.0)
        self._ml_available = False
        self._compute_fn = None

    def setup(self) -> None:
        try:
            import ssimulacra2

            self._compute_fn = ssimulacra2.compute_ssimulacra2
            self._ml_available = True
            logger.info("SSIMULACRA 2 module initialised")
        except ImportError:
            logger.warning(
                "ssimulacra2 not installed. Install with: pip install ssimulacra2"
            )
        except Exception as e:
            logger.warning(f"Failed to setup SSIMULACRA 2: {e}")

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

            # Convert BGR to RGB
            ref_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
            dist_rgb = cv2.cvtColor(dist_img, cv2.COLOR_BGR2RGB)

            score = float(self._compute_fn(ref_rgb, dist_rgb))
            return score
        except Exception as e:
            logger.debug(f"SSIMULACRA 2 scoring failed: {e}")
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

            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.ssimulacra2 = score

            if score > self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"High SSIMULACRA 2 (visible distortion): {score:.1f}",
                        details={"ssimulacra2": score, "threshold": self.warning_threshold},
                        recommendation="Significant perceptual distortion vs reference.",
                    )
                )
            logger.debug(f"SSIMULACRA 2 for {sample.path.name}: {score:.1f}")
        except Exception as e:
            logger.error(f"SSIMULACRA 2 failed for {sample.path}: {e}")
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
                ref_rgb = cv2.cvtColor(ref_f, cv2.COLOR_BGR2RGB)
                dist_rgb = cv2.cvtColor(dist_f, cv2.COLOR_BGR2RGB)
                # Align sizes
                h = min(ref_rgb.shape[0], dist_rgb.shape[0])
                w = min(ref_rgb.shape[1], dist_rgb.shape[1])
                ref_rgb = cv2.resize(ref_rgb, (w, h))
                dist_rgb = cv2.resize(dist_rgb, (w, h))
                try:
                    s = float(self._compute_fn(ref_rgb, dist_rgb))
                    scores.append(s)
                except Exception:
                    pass
            idx += 1

        ref_cap.release()
        dist_cap.release()
        return float(np.mean(scores)) if scores else None
