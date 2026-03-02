"""Compression Artifacts Detection module.

Detects blocking, ringing, and mosquito noise artifacts from video compression.
Uses frequency domain analysis (DCT) to identify compression-related distortions.
Range: 0-100 (lower = fewer artifacts, higher = more severe artifacts).
"""

import logging
from pathlib import Path

import cv2
import numpy as np

from ayase.models import Sample, ValidationIssue, ValidationSeverity, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class CompressionArtifactsModule(PipelineModule):
    name = "compression_artifacts"
    description = "Detects compression artifacts (blocking, ringing, mosquito noise)"
    default_config = {
        # Pure OpenCV/NumPy
        "subsample": 3,  # Process every Nth frame
        "warning_threshold": 40.0,  # Warn if artifact score > 40
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 3)
        self.warning_threshold = self.config.get("warning_threshold", 40.0)

    def setup(self) -> None:
        pass  # No setup needed

    def _detect_blocking(self, frame: np.ndarray) -> float:
        """Detect blocking artifacts (8x8 block boundaries).

        Args:
            frame: Input frame (H, W, C) in BGR

        Returns:
            Blocking severity (0-1)
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            h, w = gray.shape

            # Detect horizontal block boundaries
            h_diff = []
            for y in range(8, h, 8):
                if y < h - 1:
                    diff = np.abs(gray[y, :] - gray[y - 1, :]).mean()
                    h_diff.append(diff)

            # Detect vertical block boundaries
            v_diff = []
            for x in range(8, w, 8):
                if x < w - 1:
                    diff = np.abs(gray[:, x] - gray[:, x - 1]).mean()
                    v_diff.append(diff)

            if h_diff and v_diff:
                blocking = (np.mean(h_diff) + np.mean(v_diff)) / 2.0
                # Normalize (empirical max ~20)
                return min(blocking / 20.0, 1.0)

            return 0.0

        except Exception as e:
            logger.debug(f"Blocking detection failed: {e}")
            return 0.0

    def _detect_ringing(self, frame: np.ndarray) -> float:
        """Detect ringing artifacts (high-frequency oscillations near edges).

        Args:
            frame: Input frame (H, W, C) in BGR

        Returns:
            Ringing severity (0-1)
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect edges
            edges = cv2.Canny(gray, 50, 150)

            # Apply high-pass filter to detect oscillations
            kernel = np.array([[-1, -1, -1],
                             [-1,  8, -1],
                             [-1, -1, -1]])
            high_freq = cv2.filter2D(gray, -1, kernel)

            # Measure high-frequency content near edges
            edge_mask = cv2.dilate(edges, np.ones((5, 5), np.uint8))
            ringing_regions = np.abs(high_freq[edge_mask > 0])

            if len(ringing_regions) > 0:
                ringing = ringing_regions.mean() / 255.0
                return min(ringing, 1.0)

            return 0.0

        except Exception as e:
            logger.debug(f"Ringing detection failed: {e}")
            return 0.0

    def _detect_mosquito_noise(self, frame: np.ndarray) -> float:
        """Detect mosquito noise (temporal flickering near edges).

        Note: Requires multiple frames for proper detection.
        This is a simplified single-frame approximation.

        Args:
            frame: Input frame (H, W, C) in BGR

        Returns:
            Mosquito noise severity (0-1)
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect edges
            edges = cv2.Canny(gray, 50, 150)

            # Compute local variance near edges
            kernel = np.ones((3, 3), np.float32) / 9
            mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            sqmean = cv2.filter2D((gray.astype(np.float32) ** 2), -1, kernel)
            variance = sqmean - mean ** 2

            # Measure variance near edges
            edge_mask = cv2.dilate(edges, np.ones((3, 3), np.uint8))
            noise_regions = variance[edge_mask > 0]

            if len(noise_regions) > 0:
                mosquito = np.sqrt(noise_regions.mean()) / 128.0
                return min(mosquito, 1.0)

            return 0.0

        except Exception as e:
            logger.debug(f"Mosquito noise detection failed: {e}")
            return 0.0

    def _compute_artifacts_score(self, frame: np.ndarray) -> float:
        """Compute overall compression artifacts score.

        Args:
            frame: Input frame

        Returns:
            Artifacts score (0-100, higher = more artifacts)
        """
        blocking = self._detect_blocking(frame)
        ringing = self._detect_ringing(frame)
        mosquito = self._detect_mosquito_noise(frame)

        # Weighted combination
        artifacts = (blocking * 0.4 + ringing * 0.4 + mosquito * 0.2) * 100.0

        return float(artifacts)

    def process(self, sample: Sample) -> Sample:
        """Process sample to detect compression artifacts."""
        if not sample.is_video:
            # Process single image
            try:
                img = cv2.imread(str(sample.path))
                if img is not None:
                    artifacts_score = self._compute_artifacts_score(img)

                    if sample.quality_metrics is None:
                        sample.quality_metrics = QualityMetrics()
                    sample.quality_metrics.compression_artifacts = artifacts_score

                    if artifacts_score > self.warning_threshold:
                        sample.validation_issues.append(
                            ValidationIssue(
                                severity=ValidationSeverity.WARNING,
                                message=f"Compression artifacts detected: {artifacts_score:.1f}",
                                details={"artifacts_score": artifacts_score},
                                recommendation="Significant compression artifacts present. "
                                "Consider higher bitrate or less aggressive compression.",
                            )
                        )

                    logger.debug(f"Compression artifacts for {sample.path.name}: {artifacts_score:.1f}")

            except Exception as e:
                logger.warning(f"Compression artifacts detection failed for {sample.path}: {e}")

            return sample

        # Process video
        try:
            cap = cv2.VideoCapture(str(sample.path))
            artifacts_scores = []
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % self.subsample == 0:
                    score = self._compute_artifacts_score(frame)
                    artifacts_scores.append(score)

                frame_idx += 1

            cap.release()

            if not artifacts_scores:
                return sample

            avg_artifacts = np.mean(artifacts_scores)

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.compression_artifacts = avg_artifacts

            if avg_artifacts > self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Compression artifacts detected: {avg_artifacts:.1f}",
                        details={"artifacts_score": avg_artifacts},
                        recommendation="Significant compression artifacts present. "
                        "Consider higher bitrate or less aggressive compression.",
                    )
                )

            logger.debug(f"Compression artifacts for {sample.path.name}: {avg_artifacts:.1f}")

        except Exception as e:
            logger.warning(f"Compression artifacts processing failed for {sample.path}: {e}")

        return sample
