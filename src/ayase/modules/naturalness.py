"""Naturalness Score module.

Measures how "natural" vs "synthetic" video/image content appears.
Uses natural scene statistics (NSS) or learning-based approaches.
Range: 0-1 (higher = more natural).
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ayase.models import Sample, ValidationIssue, ValidationSeverity, QualityMetrics
from ayase.base_modules import NoReferenceModule

logger = logging.getLogger(__name__)


class NaturalnessModule(NoReferenceModule):
    name = "naturalness"
    description = "Measures naturalness of content (natural vs synthetic)"
    default_config = {
        "use_pyiqa": True,  # Use pyiqa's NIQE/BRISQUE as proxy
        "subsample": 2,  # Process every Nth frame
        "warning_threshold": 0.4,  # Warn if naturalness < 0.4
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.use_pyiqa = self.config.get("use_pyiqa", True)
        self.subsample = self.config.get("subsample", 2)
        self.warning_threshold = self.config.get("warning_threshold", 0.4)
        self._ml_available = False
        self._metric = None

    def setup(self) -> None:
        try:
            if self.use_pyiqa:
                import pyiqa

                # BRISQUE is a good NSS-based metric for naturalness
                self._metric = pyiqa.create_metric('brisque', device='cpu')
                self._ml_available = True
                logger.info("Naturalness module initialized with BRISQUE")
            else:
                # Fallback to manual NSS computation
                self._ml_available = True
                logger.info("Naturalness module initialized with manual NSS")

        except ImportError:
            logger.warning(
                "pyiqa not installed. Using fallback manual NSS. "
                "Install with: pip install pyiqa"
            )
            self._ml_available = True  # Can still use fallback
        except Exception as e:
            logger.warning(f"Failed to setup Naturalness: {e}")

    def _compute_mscn_features(self, gray: np.ndarray) -> np.ndarray:
        """Compute Mean Subtracted Contrast Normalized (MSCN) features.

        These are the basis of natural scene statistics.

        Args:
            gray: Grayscale image

        Returns:
            MSCN coefficients
        """
        # Gaussian kernel
        kernel = cv2.getGaussianKernel(7, 7/6)
        kernel = kernel @ kernel.T

        # Local mean
        mu = cv2.filter2D(gray.astype(np.float64), -1, kernel)

        # Local variance
        sigma = cv2.filter2D((gray.astype(np.float64) ** 2), -1, kernel)
        sigma = np.sqrt(np.abs(sigma - mu ** 2))

        # MSCN coefficients
        mscn = (gray - mu) / (sigma + 1)

        return mscn

    def _compute_nss_naturalness(self, frame: np.ndarray) -> float:
        """Compute naturalness using NSS (manual implementation).

        Args:
            frame: Input frame (H, W, C) in BGR

        Returns:
            Naturalness score (0-1, higher = more natural)
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)

            # Compute MSCN
            mscn = self._compute_mscn_features(gray)

            # Natural images have MSCN coefficients following Gaussian distribution
            # Compute shape parameter of Generalized Gaussian Distribution
            variance = np.var(mscn)
            mean_abs = np.mean(np.abs(mscn))

            if mean_abs > 0:
                # Shape parameter (beta) for GGD
                # Natural images typically have beta ~ 0.9-1.1
                beta = variance / (mean_abs ** 2)

                # Distance from ideal natural distribution
                # Ideal beta ≈ 1.0 for natural images
                naturalness = 1.0 - min(abs(beta - 1.0), 1.0)
            else:
                naturalness = 0.5

            return float(np.clip(naturalness, 0, 1))

        except Exception as e:
            logger.debug(f"NSS naturalness computation failed: {e}")
            return 0.5

    def compute_nr_score(self, sample_path: Path) -> Optional[float]:
        """Compute naturalness score.

        Args:
            sample_path: Path to video/image

        Returns:
            Naturalness score (0-1), or None if computation failed
        """
        try:
            # Check if video or image
            sample_str = str(sample_path)
            if sample_str.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                return self._compute_naturalness_video(sample_path)
            else:
                return self._compute_naturalness_image(sample_path)

        except Exception as e:
            logger.warning(f"Naturalness computation failed: {e}")
            return None

    def _compute_naturalness_image(self, sample_path: Path) -> Optional[float]:
        """Compute naturalness for single image."""
        try:
            if self._metric is not None:
                # Use pyiqa BRISQUE
                # BRISQUE returns 0-100 (lower is better)
                # Convert to naturalness score (higher is better)
                brisque_score = self._metric(str(sample_path)).item()
                # Normalize: BRISQUE ~0 is natural, ~100 is unnatural
                naturalness = 1.0 - min(brisque_score / 100.0, 1.0)
                return float(naturalness)
            else:
                # Fallback to manual NSS
                img = cv2.imread(str(sample_path))
                if img is None:
                    return None
                return self._compute_nss_naturalness(img)

        except Exception as e:
            logger.debug(f"Naturalness image computation failed: {e}")
            return None

    def _compute_naturalness_video(self, sample_path: Path) -> Optional[float]:
        """Compute naturalness for video (average across frames)."""
        try:
            cap = cv2.VideoCapture(str(sample_path))
            naturalness_scores = []
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % self.subsample == 0:
                    if self._metric is not None:
                        # Save frame temporarily for pyiqa
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                            tmp_path = tmp.name
                        cv2.imwrite(tmp_path, frame)

                        try:
                            brisque_score = self._metric(tmp_path).item()
                            naturalness = 1.0 - min(brisque_score / 100.0, 1.0)
                            naturalness_scores.append(naturalness)
                        finally:
                            Path(tmp_path).unlink(missing_ok=True)
                    else:
                        # Manual NSS
                        score = self._compute_nss_naturalness(frame)
                        naturalness_scores.append(score)

                frame_idx += 1

            cap.release()

            if not naturalness_scores:
                return None

            return float(np.mean(naturalness_scores))

        except Exception as e:
            logger.debug(f"Naturalness video computation failed: {e}")
            return None

    def process(self, sample: Sample) -> Sample:
        """Process sample with naturalness metric."""
        if not self._ml_available:
            return sample

        try:
            # Compute naturalness score
            naturalness_score = self.compute_nr_score(sample.path)

            if naturalness_score is None:
                return sample

            # Store in quality metrics
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.naturalness_score = naturalness_score

            # Add validation issue if score is low
            if naturalness_score < self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message=f"Low naturalness score: {naturalness_score:.3f}",
                        details={
                            "naturalness": naturalness_score,
                            "threshold": self.warning_threshold,
                        },
                        recommendation="Content appears synthetic or unnatural. "
                        "May indicate generated/artificial content.",
                    )
                )

            logger.debug(f"Naturalness score for {sample.path.name}: {naturalness_score:.3f}")

        except Exception as e:
            logger.warning(f"Naturalness processing failed for {sample.path}: {e}")

        return sample
