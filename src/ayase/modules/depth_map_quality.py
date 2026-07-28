"""Depth Map Quality module.

Evaluates the quality of monocular depth estimation:

  depth_quality — 0-100 (higher = better depth map quality)

Aspects assessed:
  - Sharpness: depth map edge crispness (Laplacian variance)
  - Completeness: percentage of valid (non-zero) depth pixels
  - Discontinuity preservation: edges in RGB should align with
    depth edges (gradient correlation)

Uses MiDaS (small) for depth estimation, then evaluates the
quality of the resulting depth map against the RGB image.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

_MIDAS_MIRRORS = {
    "MiDaS_small": (
        "midas_v21_small_256.pt",
        "depth_map_quality/midas_v21_small_256.pt",
    ),
    "DPT_Hybrid": ("dpt_hybrid_384.pt", "depth_map_quality/dpt_hybrid_384.pt"),
    "DPT_Large": ("dpt_large_384.pt", "depth_map_quality/dpt_large_384.pt"),
}
_MIRROR_BASE = "https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/main/"


class DepthMapQualityModule(PipelineModule):
    name = "depth_map_quality"
    description = "Monocular depth map quality (sharpness, completeness, edge alignment)"
    default_config = {
        "model_type": "MiDaS_small",
        "device": "auto",
        "subsample": 10,
        "max_frames": 30,
    }
    metric_groups = {
        "depth_quality": "spatial",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.model_type = self.config.get("model_type", "MiDaS_small")
        self.device_config = self.config.get("device", "auto")
        self.subsample = self.config.get("subsample", 10)
        self.max_frames = self.config.get("max_frames", 30)

        self.device = None
        self._model = None
        self._transform = None
        self._ml_available = False
        self._backend = "unavailable"

    def setup(self) -> None:
        try:
            import torch
            import os
            from ayase.config import download_torch_hub_checkpoint
            from ayase.runtime import resolve_torch_device

            self.device = torch.device(resolve_torch_device(self.device_config))
            models_dir = str(self.config.get("models_dir", "models"))
            os.environ["TORCH_HOME"] = models_dir
            mirror = _MIDAS_MIRRORS.get(self.model_type)
            if mirror is None:
                raise ValueError(f"Unsupported mirrored MiDaS model: {self.model_type}")
            filename, relative_path = mirror
            checkpoint_path = download_torch_hub_checkpoint(
                filename, _MIRROR_BASE + relative_path, models_dir
            )

            from ayase.modules._midas_utils import load_midas_model

            self._model = load_midas_model(self.model_type, checkpoint_path, self.device)

            midas_transforms = torch.hub.load(
                "intel-isl/MiDaS", "transforms", trust_repo=True
            )
            if "small" in self.model_type.lower():
                self._transform = midas_transforms.small_transform
            else:
                self._transform = midas_transforms.dpt_transform

            self._ml_available = True
            self._backend = f"midas:{self.model_type}"
            logger.info(f"Depth map quality: {self.model_type} on {self.device}")

        except Exception as e:
            logger.warning(f"Failed to setup depth map quality: {e}")

    def _estimate_depth(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Return depth map as float32 array."""
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

            return prediction.cpu().numpy().astype(np.float32)
        except Exception as e:
            logger.debug(f"Depth estimation failed: {e}")
            return None

    @staticmethod
    def _evaluate_depth(frame_bgr: np.ndarray, depth: np.ndarray) -> float:
        """Evaluate depth map quality against RGB image."""
        # 1. Sharpness: Laplacian variance of depth map
        d_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        lap_var = float(cv2.Laplacian(d_norm, cv2.CV_64F).var())
        sharpness = min(lap_var / 100.0, 1.0)

        # 2. Completeness: fraction of non-zero pixels
        completeness = float(np.count_nonzero(depth) / max(depth.size, 1))

        # 3. Edge alignment: correlation between RGB edges and depth edges
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        rgb_edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
        depth_edges = cv2.Canny(d_norm, 30, 100).astype(np.float32) / 255.0

        # Correlation via overlap
        if rgb_edges.sum() > 0:
            overlap = float((rgb_edges * depth_edges).sum() / rgb_edges.sum())
        else:
            overlap = 1.0

        # Combined score
        score = (0.40 * sharpness + 0.25 * completeness + 0.35 * overlap) * 100
        return float(np.clip(score, 0, 100))

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            if sample.is_video:
                score = self._process_video(sample.path)
            else:
                score = self._process_image(sample.path)

            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.depth_quality = score

            logger.debug(f"Depth quality for {sample.path.name}: {score:.1f}")

        except Exception as e:
            logger.error(f"Depth map quality failed for {sample.path}: {e}")

        return sample

    def _process_image(self, path: Path) -> Optional[float]:
        img = cv2.imread(str(path))
        if img is None:
            return None
        depth = self._estimate_depth(img)
        if depth is None:
            return None
        return self._evaluate_depth(img, depth)

    def _process_video(self, path: Path) -> Optional[float]:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None

        scores = []
        idx = 0

        try:
            while idx < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % self.subsample == 0:
                    depth = self._estimate_depth(frame)
                    if depth is not None:
                        scores.append(self._evaluate_depth(frame, depth))
                idx += 1
        finally:
            cap.release()
        return float(np.mean(scores)) if scores else None
