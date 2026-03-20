"""DAVIS J&F — Video Object Segmentation Quality (DAVIS 2016).

Full-reference metric for evaluating video segmentation quality:
  J (Jaccard / IoU): region-based accuracy of predicted masks
  F (F-measure): contour-based accuracy of predicted masks

Expects reference segmentation masks. Uses heuristic based on
mask IoU and boundary F-measure computation.

davis_j — 0-1, higher = better (region IoU)
davis_f — 0-1, higher = better (boundary F-measure)
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.base_modules import ReferenceBasedModule

logger = logging.getLogger(__name__)


class DAVISJFModule(ReferenceBasedModule):
    name = "davis_jf"
    description = "DAVIS J&F video segmentation quality (FR, 2016)"
    metric_field = None  # We override process() to set both davis_j and davis_f
    default_config = {"subsample": 8, "boundary_threshold": 2}

    def __init__(self, config=None):
        super().__init__(config)
        self._model = None
        self.subsample = self.config.get("subsample", 8)
        self.boundary_threshold = self.config.get("boundary_threshold", 2)

    def setup(self) -> None:
        logger.info("DAVIS J&F module initialised (heuristic)")

    def compute_reference_score(self, sample_path: Path, reference_path: Path) -> Optional[float]:
        """Not used directly; process() is overridden instead."""
        return None

    def process(self, sample: Sample) -> Sample:
        """Override to set both davis_j and davis_f."""
        reference = getattr(sample, "reference_path", None)
        if reference is None:
            return sample

        if not isinstance(reference, Path):
            reference = Path(reference)
        if not reference.exists():
            return sample

        try:
            if str(sample.path).lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                scores = self._score_video(str(sample.path), str(reference))
            else:
                scores = self._score_image(str(sample.path), str(reference))

            if scores is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.davis_j = scores["j"]
            sample.quality_metrics.davis_f = scores["f"]
            logger.debug(
                f"DAVIS J&F for {sample.path.name}: "
                f"J={scores['j']:.4f} F={scores['f']:.4f}"
            )
        except Exception as e:
            logger.warning(f"DAVIS J&F failed: {e}")
        return sample

    def _score_image(self, sample_p: str, ref_p: str) -> Optional[dict]:
        pred = cv2.imread(sample_p, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(ref_p, cv2.IMREAD_GRAYSCALE)
        if pred is None or gt is None:
            return None

        h, w = gt.shape[:2]
        pred = cv2.resize(pred, (w, h))

        pred_mask = (pred > 127).astype(np.uint8)
        gt_mask = (gt > 127).astype(np.uint8)

        j = self._compute_jaccard(pred_mask, gt_mask)
        f = self._compute_boundary_f(pred_mask, gt_mask)
        return {"j": j, "f": f}

    def _score_video(self, sample_p: str, ref_p: str) -> Optional[dict]:
        cap_s = cv2.VideoCapture(sample_p)
        cap_r = cv2.VideoCapture(ref_p)
        try:
            total = min(
                int(cap_s.get(cv2.CAP_PROP_FRAME_COUNT)),
                int(cap_r.get(cv2.CAP_PROP_FRAME_COUNT)),
            )
            if total <= 0:
                return None

            indices = np.linspace(0, total - 1, min(self.subsample, total), dtype=int)
            j_scores = []
            f_scores = []

            for idx in indices:
                cap_s.set(cv2.CAP_PROP_POS_FRAMES, idx)
                cap_r.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret_s, frame_s = cap_s.read()
                ret_r, frame_r = cap_r.read()
                if not (ret_s and ret_r):
                    continue

                pred = cv2.cvtColor(frame_s, cv2.COLOR_BGR2GRAY) if frame_s.ndim == 3 else frame_s
                gt = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY) if frame_r.ndim == 3 else frame_r

                h, w = gt.shape[:2]
                pred = cv2.resize(pred, (w, h))

                pred_mask = (pred > 127).astype(np.uint8)
                gt_mask = (gt > 127).astype(np.uint8)

                j_scores.append(self._compute_jaccard(pred_mask, gt_mask))
                f_scores.append(self._compute_boundary_f(pred_mask, gt_mask))

            if not j_scores:
                return None

            return {
                "j": float(np.mean(j_scores)),
                "f": float(np.mean(f_scores)),
            }
        finally:
            cap_s.release()
            cap_r.release()

    def _compute_jaccard(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Compute Jaccard index (IoU) between binary masks."""
        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        if union == 0:
            return 1.0  # Both empty = perfect match
        return float(intersection / union)

    def _compute_boundary_f(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Compute boundary F-measure between binary masks.

        Extracts contours and computes precision/recall based on boundary
        pixel distance.
        """
        # Extract boundaries
        pred_boundary = self._get_boundary(pred)
        gt_boundary = self._get_boundary(gt)

        if pred_boundary.sum() == 0 and gt_boundary.sum() == 0:
            return 1.0  # Both have no boundaries
        if pred_boundary.sum() == 0 or gt_boundary.sum() == 0:
            return 0.0

        # Distance transform for tolerance matching
        from cv2 import distanceTransform, DIST_L2

        gt_dist = distanceTransform(1 - gt_boundary, DIST_L2, 3)
        pred_dist = distanceTransform(1 - pred_boundary, DIST_L2, 3)

        # Precision: predicted boundary pixels within threshold of GT boundary
        precision = float(
            np.sum(pred_boundary * (gt_dist <= self.boundary_threshold)) /
            (pred_boundary.sum() + 1e-8)
        )

        # Recall: GT boundary pixels within threshold of predicted boundary
        recall = float(
            np.sum(gt_boundary * (pred_dist <= self.boundary_threshold)) /
            (gt_boundary.sum() + 1e-8)
        )

        if precision + recall < 1e-8:
            return 0.0
        f_measure = 2 * precision * recall / (precision + recall)
        return float(np.clip(f_measure, 0.0, 1.0))

    def _get_boundary(self, mask: np.ndarray) -> np.ndarray:
        """Extract boundary pixels from a binary mask using morphological erosion."""
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(mask, kernel, iterations=1)
        boundary = (mask - eroded).astype(np.uint8)
        return boundary
