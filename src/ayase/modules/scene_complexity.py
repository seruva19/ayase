"""Scene Complexity module.

Measures spatial and temporal complexity of video content.
Higher complexity indicates more detailed/dynamic scenes.
Range: 0-100 (higher = more complex).
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class SceneComplexityModule(PipelineModule):
    name = "scene_complexity"
    description = "Spatial and temporal scene complexity analysis"
    default_config = {
        # Pure OpenCV, no ML needed
        "subsample": 2,  # Process every Nth frame
        "spatial_weight": 0.5,  # Weight for spatial complexity
        "temporal_weight": 0.5,  # Weight for temporal complexity
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 2)
        self.spatial_weight = self.config.get("spatial_weight", 0.5)
        self.temporal_weight = self.config.get("temporal_weight", 0.5)

    def setup(self) -> None:
        # No setup needed, pure OpenCV
        pass

    def _compute_spatial_complexity(self, frame: np.ndarray) -> float:
        """Compute spatial complexity of a frame.

        Measures edge density and color diversity.

        Args:
            frame: Input frame (H, W, C) in BGR

        Returns:
            Spatial complexity score (0-1)
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 1. Edge density (Canny)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size

            # 2. Color diversity (HSV histogram entropy)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            hist_h = hist_h / hist_h.sum()
            hist_h = hist_h[hist_h > 0]  # Remove zeros
            entropy_h = -np.sum(hist_h * np.log2(hist_h + 1e-10))

            # Normalize entropy (max entropy for 180 bins)
            max_entropy = np.log2(180)
            color_diversity = entropy_h / max_entropy

            # 3. Texture complexity (standard deviation)
            texture = gray.std() / 128.0  # Normalize to [0, 1]

            # Combine metrics
            spatial_complexity = (edge_density + color_diversity + texture) / 3.0

            return float(np.clip(spatial_complexity, 0, 1))

        except Exception as e:
            logger.debug(f"Spatial complexity computation failed: {e}")
            return 0.5

    def _compute_temporal_complexity(self, frames: list) -> float:
        """Compute temporal complexity from frame sequence.

        Measures frame differences and motion.

        Args:
            frames: List of consecutive frames

        Returns:
            Temporal complexity score (0-1)
        """
        if len(frames) < 2:
            return 0.0

        try:
            # 1. Frame-to-frame difference
            frame_diffs = []
            for i in range(len(frames) - 1):
                diff = cv2.absdiff(frames[i], frames[i + 1])
                diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                frame_diffs.append(diff_gray.mean())

            avg_diff = np.mean(frame_diffs) / 255.0  # Normalize

            # 2. Motion magnitude (optical flow)
            try:
                gray1 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2GRAY)

                flow = cv2.calcOpticalFlowFarneback(
                    gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                motion = magnitude.mean() / 10.0  # Normalize (empirical max ~10)

            except Exception:
                motion = 0.0

            # Combine metrics
            temporal_complexity = (avg_diff + motion) / 2.0

            return float(np.clip(temporal_complexity, 0, 1))

        except Exception as e:
            logger.debug(f"Temporal complexity computation failed: {e}")
            return 0.5

    def process(self, sample: Sample) -> Sample:
        """Process sample to compute scene complexity."""
        if not sample.is_video:
            # For images, only spatial complexity
            try:
                img = cv2.imread(str(sample.path))
                if img is not None:
                    spatial = self._compute_spatial_complexity(img)
                    complexity = spatial * 100.0  # Scale to 0-100

                    if sample.quality_metrics is None:
                        sample.quality_metrics = QualityMetrics()
                    sample.quality_metrics.scene_complexity = complexity

                    logger.debug(f"Scene complexity for {sample.path.name}: {complexity:.1f}")

            except Exception as e:
                logger.warning(f"Scene complexity failed for {sample.path}: {e}")

            return sample

        # For videos, compute both spatial and temporal
        try:
            cap = cv2.VideoCapture(str(sample.path))
            frames = []
            spatial_scores = []
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % self.subsample == 0:
                    # Compute spatial complexity
                    spatial = self._compute_spatial_complexity(frame)
                    spatial_scores.append(spatial)

                    # Keep frames for temporal analysis
                    frames.append(frame)
                    if len(frames) > 10:  # Keep last 10 frames
                        frames.pop(0)

                frame_idx += 1

            cap.release()

            if not spatial_scores:
                return sample

            # Average spatial complexity
            avg_spatial = np.mean(spatial_scores)

            # Compute temporal complexity
            temporal = self._compute_temporal_complexity(frames)

            # Combined complexity score
            complexity = (
                avg_spatial * self.spatial_weight + temporal * self.temporal_weight
            ) * 100.0

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.scene_complexity = complexity

            logger.debug(
                f"Scene complexity for {sample.path.name}: {complexity:.1f} "
                f"(spatial: {avg_spatial*100:.1f}, temporal: {temporal*100:.1f})"
            )

        except Exception as e:
            logger.warning(f"Scene complexity processing failed for {sample.path}: {e}")

        return sample
