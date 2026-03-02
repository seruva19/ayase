"""Monocular Depth Consistency module.

Measures temporal coherence of monocular depth estimation across
video frames:

  depth_temporal_consistency — 0-1 (higher = more consistent)

Algorithm:
  1. Estimate per-frame depth using MiDaS (small model) or DPT.
  2. Compute frame-to-frame depth correlation.
  3. Detect depth flickering (sudden inversions / large jumps).

Videos only — images are skipped (single frame has no temporal signal).
"""

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class DepthConsistencyModule(PipelineModule):
    name = "depth_consistency"
    description = "Monocular depth temporal consistency"
    default_config = {
        "model_type": "MiDaS_small",  # "MiDaS_small", "DPT_Hybrid", "DPT_Large"
        "device": "auto",
        "subsample": 3,
        "max_frames": 200,
        "warning_threshold": 0.7,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.model_type = self.config.get("model_type", "MiDaS_small")
        self.device_config = self.config.get("device", "auto")
        self.subsample = self.config.get("subsample", 3)
        self.max_frames = self.config.get("max_frames", 200)
        self.warning_threshold = self.config.get("warning_threshold", 0.7)

        self.device = None
        self._model = None
        self._transform = None
        self._ml_available = False

    def setup(self) -> None:
        try:
            import torch

            if self.device_config == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(self.device_config)

            # Load MiDaS via torch hub
            self._model = torch.hub.load(
                "intel-isl/MiDaS", self.model_type, trust_repo=True
            )
            self._model.to(self.device).eval()

            midas_transforms = torch.hub.load(
                "intel-isl/MiDaS", "transforms", trust_repo=True
            )

            if "small" in self.model_type.lower():
                self._transform = midas_transforms.small_transform
            elif "DPT" in self.model_type:
                self._transform = midas_transforms.dpt_transform
            else:
                self._transform = midas_transforms.small_transform

            self._ml_available = True
            logger.info(
                f"Depth consistency: {self.model_type} initialised on {self.device}"
            )

        except Exception as e:
            logger.warning(f"Failed to setup depth consistency: {e}")

    def _estimate_depth(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Return a normalised depth map (H, W) in [0, 1]."""
        try:
            import torch

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            input_batch = self._transform(rgb).to(self.device)

            with torch.no_grad():
                prediction = self._model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=rgb.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            depth = prediction.cpu().numpy()

            # Normalise to 0-1
            d_min, d_max = depth.min(), depth.max()
            if d_max - d_min > 1e-6:
                depth = (depth - d_min) / (d_max - d_min)
            else:
                depth = np.zeros_like(depth)

            return depth.astype(np.float32)

        except Exception as e:
            logger.debug(f"Depth estimation failed: {e}")
            return None

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or not sample.is_video:
            return sample

        try:
            return self._process_video(sample)
        except Exception as e:
            logger.error(f"Depth consistency failed for {sample.path}: {e}")
            return sample

    def _process_video(self, sample: Sample) -> Sample:
        cap = cv2.VideoCapture(str(sample.path))
        if not cap.isOpened():
            return sample

        depth_maps: List[np.ndarray] = []
        idx = 0

        while idx < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % self.subsample == 0:
                dm = self._estimate_depth(frame)
                if dm is not None:
                    depth_maps.append(dm)
            idx += 1

        cap.release()

        if len(depth_maps) < 3:
            return sample

        # Compute temporal consistency
        correlations = []
        for i in range(1, len(depth_maps)):
            prev = depth_maps[i - 1].flatten()
            curr = depth_maps[i].flatten()
            # Pearson correlation
            corr = float(np.corrcoef(prev, curr)[0, 1])
            if not np.isnan(corr):
                correlations.append(corr)

        if not correlations:
            return sample

        consistency = float(np.mean(correlations))
        consistency = float(np.clip(consistency, 0, 1))

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()

        sample.quality_metrics.depth_temporal_consistency = consistency

        if consistency < self.warning_threshold:
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Low depth consistency: {consistency:.3f}",
                    details={
                        "depth_temporal_consistency": consistency,
                        "min_correlation": float(min(correlations)),
                    },
                    recommendation=(
                        "Depth maps flicker between frames. "
                        "May indicate 3D structure inconsistency."
                    ),
                )
            )

        logger.debug(
            f"Depth consistency for {sample.path.name}: {consistency:.3f} "
            f"(min={min(correlations):.3f})"
        )

        return sample
