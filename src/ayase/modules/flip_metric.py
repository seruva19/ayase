"""NVIDIA FLIP module.

FLIP (Finding Local Interesting Points) is NVIDIA's perceptual image
quality metric designed for comparing rendered images. It models human
perception of differences between reference and test images.

Supports both LDR-FLIP and HDR-FLIP variants.

Range: 0-1 (lower = less perceptual difference, 0 = identical).

This is a full-reference metric.

Requires the ``flip-evaluator`` or ``flip-torch`` Python package:
  pip install flip-evaluator
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics, ValidationIssue, ValidationSeverity
from ayase.base_modules import ReferenceBasedModule

logger = logging.getLogger(__name__)


class FLIPModule(ReferenceBasedModule):
    name = "flip"
    description = "NVIDIA FLIP perceptual difference (0-1, lower=better)"
    default_config = {
        "subsample": 5,
        "warning_threshold": 0.3,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 5)
        self.warning_threshold = self.config.get("warning_threshold", 0.3)
        self._ml_available = False
        self._backend = None

    def setup(self) -> None:
        # Try flip-evaluator package
        try:
            import flip_evaluator

            self._backend = "flip_evaluator"
            self._ml_available = True
            logger.info("FLIP module initialised (flip-evaluator backend)")
            return
        except ImportError:
            pass

        # Try flip_torch
        try:
            import flip_torch

            self._backend = "flip_torch"
            self._ml_available = True
            logger.info("FLIP module initialised (flip-torch backend)")
            return
        except ImportError:
            pass

        # Fallback: OpenCV approximation using multi-scale colour + edge
        self._backend = "approx"
        self._ml_available = True
        logger.info("FLIP module initialised (OpenCV approximation fallback)")

    def compute_reference_score(
        self, sample_path: Path, reference_path: Path
    ) -> Optional[float]:
        try:
            ref_img = cv2.imread(str(reference_path))
            dist_img = cv2.imread(str(sample_path))
            if ref_img is None or dist_img is None:
                return None

            h = min(ref_img.shape[0], dist_img.shape[0])
            w = min(ref_img.shape[1], dist_img.shape[1])
            ref_img = cv2.resize(ref_img, (w, h))
            dist_img = cv2.resize(dist_img, (w, h))

            if self._backend == "flip_evaluator":
                return self._compute_flip_evaluator(ref_img, dist_img)
            elif self._backend == "flip_torch":
                return self._compute_flip_torch(ref_img, dist_img)
            else:
                return self._compute_approx(ref_img, dist_img)
        except Exception as e:
            logger.debug(f"FLIP scoring failed: {e}")
            return None

    def _compute_flip_evaluator(self, ref_bgr: np.ndarray, dist_bgr: np.ndarray) -> float:
        import flip_evaluator

        ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        dist_rgb = cv2.cvtColor(dist_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        result = flip_evaluator.evaluate(ref_rgb, dist_rgb)
        if isinstance(result, dict):
            return float(result.get("mean", result.get("flip_mean", 0.0)))
        return float(result)

    def _compute_flip_torch(self, ref_bgr: np.ndarray, dist_bgr: np.ndarray) -> float:
        import torch
        import flip_torch

        ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        dist_rgb = cv2.cvtColor(dist_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        ref_t = torch.from_numpy(ref_rgb).permute(2, 0, 1).unsqueeze(0)
        dist_t = torch.from_numpy(dist_rgb).permute(2, 0, 1).unsqueeze(0)
        flip_map = flip_torch.compute_flip(ref_t, dist_t)
        return float(flip_map.mean().item())

    def _compute_approx(self, ref_bgr: np.ndarray, dist_bgr: np.ndarray) -> float:
        """Approximate FLIP using colour difference + edge detection.

        FLIP's core idea: combine colour difference with feature difference.
        This approximation uses CIE LAB colour distance + Sobel edge difference.
        """
        ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        dist_lab = cv2.cvtColor(dist_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Colour difference (Delta E approximation, normalised)
        colour_diff = np.sqrt(np.sum((ref_lab - dist_lab) ** 2, axis=2))
        colour_score = np.clip(colour_diff / 100.0, 0, 1)

        # Edge/feature difference
        ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        dist_gray = cv2.cvtColor(dist_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        ref_edges = cv2.Sobel(ref_gray, cv2.CV_32F, 1, 1)
        dist_edges = cv2.Sobel(dist_gray, cv2.CV_32F, 1, 1)
        edge_diff = np.abs(ref_edges - dist_edges)
        edge_max = max(np.max(np.abs(ref_edges)), 1.0)
        edge_score = np.clip(edge_diff / edge_max, 0, 1)

        # FLIP-like combination: max of colour and feature
        flip_map = np.maximum(colour_score, edge_score)
        return float(np.mean(flip_map))

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

            sample.quality_metrics.flip_score = score

            if score > self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"High FLIP perceptual difference: {score:.3f}",
                        details={"flip_score": score, "threshold": self.warning_threshold},
                        recommendation="Visible perceptual differences detected.",
                    )
                )
            logger.debug(f"FLIP for {sample.path.name}: {score:.3f}")
        except Exception as e:
            logger.error(f"FLIP failed for {sample.path}: {e}")
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
                    if self._backend == "flip_evaluator":
                        s = self._compute_flip_evaluator(ref_r, dist_r)
                    elif self._backend == "flip_torch":
                        s = self._compute_flip_torch(ref_r, dist_r)
                    else:
                        s = self._compute_approx(ref_r, dist_r)
                    scores.append(s)
                except Exception:
                    pass
            idx += 1

        ref_cap.release()
        dist_cap.release()
        return float(np.mean(scores)) if scores else None


class FLIPCompatModule(FLIPModule):
    """Compatibility alias matching filename-based discovery."""

    name = "flip_metric"
