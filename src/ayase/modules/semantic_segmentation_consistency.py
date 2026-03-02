"""Semantic Segmentation Consistency module.

Measures temporal stability of semantic segmentation across video
frames:

  semantic_consistency — 0-1 (higher = more consistent)

Algorithm:
  1. Produce per-frame semantic segmentation (SegFormer or simple
     colour-based clustering as fallback).
  2. Compute frame-to-frame segment overlap (IoU per class).
  3. Average IoU across frames and classes.

Videos only — images are skipped.
"""

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class SemanticSegmentationConsistencyModule(PipelineModule):
    name = "semantic_segmentation_consistency"
    description = "Temporal stability of semantic segmentation"
    default_config = {
        "backend": "auto",  # "segformer", "kmeans", or "auto"
        "device": "auto",
        "subsample": 3,
        "max_frames": 150,
        "num_clusters": 8,  # For K-means fallback
        "warning_threshold": 0.6,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.backend = self.config.get("backend", "auto")
        self.device_config = self.config.get("device", "auto")
        self.subsample = self.config.get("subsample", 3)
        self.max_frames = self.config.get("max_frames", 150)
        self.num_clusters = self.config.get("num_clusters", 8)
        self.warning_threshold = self.config.get("warning_threshold", 0.6)

        self.device = None
        self._segformer_model = None
        self._segformer_processor = None
        self._use_segformer = False
        self._ml_available = False

    def setup(self) -> None:
        if self.backend in ("segformer", "auto"):
            try:
                import torch
                from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

                if self.device_config == "auto":
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
                else:
                    self.device = self.device_config

                self._segformer_processor = SegformerImageProcessor.from_pretrained(
                    "nvidia/segformer-b0-finetuned-ade-512-512"
                )
                self._segformer_model = SegformerForSemanticSegmentation.from_pretrained(
                    "nvidia/segformer-b0-finetuned-ade-512-512"
                ).to(self.device).eval()

                self._use_segformer = True
                self._ml_available = True
                logger.info(f"Semantic consistency: SegFormer initialised on {self.device}")
                return

            except ImportError:
                if self.backend == "segformer":
                    logger.warning("transformers not installed for SegFormer")
                    return
                logger.info("SegFormer unavailable, falling back to K-means")
            except Exception as e:
                logger.warning(f"SegFormer init failed: {e}")
                if self.backend == "segformer":
                    return

        # K-means fallback (always available)
        self._ml_available = True
        logger.info("Semantic consistency: K-means colour clustering backend")

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------

    def _segment_segformer(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Return integer label map (H, W) using SegFormer."""
        import torch

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        from PIL import Image
        pil_img = Image.fromarray(rgb)

        inputs = self._segformer_processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._segformer_model(**inputs)
            logits = outputs.logits  # (1, C, H', W')

        # Upsample to original size
        h, w = frame_bgr.shape[:2]
        upsampled = torch.nn.functional.interpolate(
            logits, size=(h, w), mode="bilinear", align_corners=False
        )
        labels = upsampled.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)
        return labels

    def _segment_kmeans(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return integer label map (H, W) using K-means colour clustering."""
        h, w = frame_bgr.shape[:2]

        # Downsample for speed
        small = cv2.resize(frame_bgr, (w // 2, h // 2))
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
        pixels = lab.reshape(-1, 3)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels_flat, _ = cv2.kmeans(
            pixels, self.num_clusters, None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )

        labels = labels_flat.reshape(h // 2, w // 2).astype(np.int32)
        # Upsample back
        labels = cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)
        return labels

    def _segment(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        try:
            if self._use_segformer:
                return self._segment_segformer(frame_bgr)
            return self._segment_kmeans(frame_bgr)
        except Exception as e:
            logger.debug(f"Segmentation failed: {e}")
            return None

    # ------------------------------------------------------------------
    # IoU computation
    # ------------------------------------------------------------------

    @staticmethod
    def _mean_iou(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
        """Compute mean IoU between two label maps."""
        classes = set(np.unique(labels_a)) | set(np.unique(labels_b))
        if not classes:
            return 1.0

        ious = []
        for c in classes:
            mask_a = labels_a == c
            mask_b = labels_b == c
            intersection = np.logical_and(mask_a, mask_b).sum()
            union = np.logical_or(mask_a, mask_b).sum()
            if union > 0:
                ious.append(float(intersection / union))

        return float(np.mean(ious)) if ious else 0.0

    # ------------------------------------------------------------------
    # Process
    # ------------------------------------------------------------------

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or not sample.is_video:
            return sample

        try:
            return self._process_video(sample)
        except Exception as e:
            logger.error(f"Semantic consistency failed for {sample.path}: {e}")
            return sample

    def _process_video(self, sample: Sample) -> Sample:
        cap = cv2.VideoCapture(str(sample.path))
        if not cap.isOpened():
            return sample

        label_maps: List[np.ndarray] = []
        idx = 0

        while idx < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % self.subsample == 0:
                labels = self._segment(frame)
                if labels is not None:
                    label_maps.append(labels)
            idx += 1

        cap.release()

        if len(label_maps) < 2:
            return sample

        # Compute frame-to-frame mean IoU
        ious = []
        for i in range(1, len(label_maps)):
            iou = self._mean_iou(label_maps[i - 1], label_maps[i])
            ious.append(iou)

        if not ious:
            return sample

        consistency = float(np.mean(ious))

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()

        sample.quality_metrics.semantic_consistency = consistency

        if consistency < self.warning_threshold:
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Low semantic consistency: {consistency:.3f}",
                    details={
                        "semantic_consistency": consistency,
                        "min_iou": float(min(ious)),
                    },
                    recommendation=(
                        "Semantic segmentation is unstable between frames. "
                        "Object boundaries may be flickering."
                    ),
                )
            )

        logger.debug(
            f"Semantic consistency for {sample.path.name}: "
            f"{consistency:.3f} (min={min(ious):.3f})"
        )

        return sample
