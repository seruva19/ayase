"""DAVIS J&F — Video Object Segmentation Quality (DAVIS 2016).

Full-reference metric for evaluating video segmentation quality:
  J (Jaccard / IoU): region-based accuracy of predicted masks
  F (F-measure): contour-based accuracy of predicted masks

Expects reference segmentation masks. Computes the DAVIS J and F measures
directly: J is the region Jaccard index (IoU) between predicted and
reference masks; F is the boundary F-measure with distance-transform
tolerance matching (the standard DAVIS toolkit formulation).

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
    provenance = "published"
    sources = {
        "davis_f": "DAVIS J&F (Perazzi et al., CVPR 2016) — db_eval_boundary, https://github.com/davisvideochallenge/davis2017-evaluation/blob/master/davis2017/metrics.py",
        "davis_j": "DAVIS J&F (Perazzi et al., CVPR 2016) — db_eval_iou, https://github.com/davisvideochallenge/davis2017-evaluation/blob/master/davis2017/metrics.py",
    }
    deviations = {
        "davis_f": "masks are read from a video container (lossy encoding may shift boundaries)",
    }
    description = "DAVIS J&F video segmentation quality (FR, 2016)"
    metric_field = None  # We override process() to set both davis_j and davis_f
    default_config = {}
    metric_groups = {
        "davis_f": "temporal",
        "davis_j": "temporal",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._model = None
        self._backend = "algorithmic"

    def setup(self) -> None:
        logger.info("DAVIS J&F module initialised (algorithmic J & F)")

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

            j_scores = []
            f_scores = []

            # Official protocol evaluates every frame — read sequentially.
            while True:
                ret_s, frame_s = cap_s.read()
                ret_r, frame_r = cap_r.read()
                if not (ret_s and ret_r):
                    break

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
        """Boundary F-measure following the official ``db_eval_boundary``.

        Boundaries come from ``seg2bmap`` (pixels whose 8-neighbourhood holds
        another label, both sides of the edge) and matching uses dilation by a
        disk of ``ceil(0.008 * diagonal)`` pixels — the toolkit tolerance.
        """
        pred_b = self._seg2bmap(pred.astype(bool))
        gt_b = self._seg2bmap(gt.astype(bool))

        bound_pix = int(np.ceil(0.008 * float(np.linalg.norm(pred.shape))))
        pred_dil = self._dilate_disk(pred_b, bound_pix)
        gt_dil = self._dilate_disk(gt_b, bound_pix)

        n_pred, n_gt = pred_b.sum(), gt_b.sum()
        # Official edge cases: empty-vs-nonempty gives (1, 0) or (0, 1).
        if n_pred == 0 and n_gt == 0:
            precision = recall = 1.0
        elif n_pred == 0:
            precision, recall = 1.0, 0.0
        elif n_gt == 0:
            precision, recall = 0.0, 1.0
        else:
            precision = float((pred_b & gt_dil).sum()) / float(n_pred)
            recall = float((gt_b & pred_dil).sum()) / float(n_gt)

        if precision + recall == 0:
            return 0.0
        return float(2 * precision * recall / (precision + recall))

    @staticmethod
    def _seg2bmap(mask: np.ndarray) -> np.ndarray:
        """Pixels whose 8-neighbourhood contains a different label."""
        padded = np.pad(mask, 1, mode="edge")
        boundary = np.zeros(mask.shape, dtype=bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                boundary |= mask != padded[1 + dy:1 + dy + mask.shape[0],
                                          1 + dx:1 + dx + mask.shape[1]]
        return boundary

    @staticmethod
    def _dilate_disk(bmap: np.ndarray, radius: int) -> np.ndarray:
        r = max(int(radius), 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        return cv2.dilate(bmap.astype(np.uint8), kernel).astype(bool)
