"""NIQE (Natural Image Quality Evaluator) module.

NIQE is a no-reference image quality metric based on natural scene statistics.
It compares image statistics to a pre-trained model of natural images.
Lower scores = better quality. Typical ranges: 2-10 (lower is better).
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ayase.models import Sample, ValidationIssue, ValidationSeverity, QualityMetrics
from ayase.base_modules import NoReferenceModule

logger = logging.getLogger(__name__)


class NIQEModule(NoReferenceModule):
    name = "niqe"
    description = "Natural Image Quality Evaluator (no-reference)"
    default_config = {
        "subsample": 2,  # Process every Nth frame for videos
        "warning_threshold": 7.0,  # Warn if NIQE > 7.0 (lower is better)
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 2)
        self.warning_threshold = self.config.get("warning_threshold", 7.0)
        self._ml_available = False
        self._niqe_metric = None

    def setup(self) -> None:
        try:
            import pyiqa

            # Create NIQE metric
            self._niqe_metric = pyiqa.create_metric('niqe', device='cpu')
            self._ml_available = True
            logger.info("NIQE module initialized")

        except ImportError:
            logger.warning("pyiqa package not installed. Install with: pip install pyiqa")
        except Exception as e:
            logger.warning(f"Failed to setup NIQE: {e}")

    def compute_nr_score(self, sample_path: Path) -> Optional[float]:
        """Compute NIQE score for sample.

        Args:
            sample_path: Path to video/image

        Returns:
            NIQE score (lower is better), or None if computation failed
        """
        try:
            # Check if video or image
            sample_str = str(sample_path)
            if sample_str.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                return self._compute_niqe_video(sample_path)
            else:
                return self._compute_niqe_image(sample_path)

        except Exception as e:
            logger.warning(f"NIQE computation failed: {e}")
            return None

    def _compute_niqe_image(self, sample_path: Path) -> Optional[float]:
        """Compute NIQE for a single image."""
        try:
            # Load image
            img = cv2.imread(str(sample_path))
            if img is None:
                return None

            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Compute NIQE
            # pyiqa expects PIL Image or path
            score = self._niqe_metric(str(sample_path)).item()

            return float(score)

        except Exception as e:
            logger.debug(f"NIQE image computation failed: {e}")
            return None

    def _compute_niqe_video(self, sample_path: Path) -> Optional[float]:
        """Compute NIQE for video (average across frames)."""
        try:
            cap = cv2.VideoCapture(str(sample_path))
            try:
                niqe_scores = []
                frame_idx = 0

                # Create temporary directory for frames
                import tempfile
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmpdir_path = Path(tmpdir)

                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        # Subsample frames
                        if frame_idx % self.subsample != 0:
                            frame_idx += 1
                            continue

                        # Save frame temporarily
                        frame_path = tmpdir_path / f"frame_{frame_idx}.png"
                        cv2.imwrite(str(frame_path), frame)

                        # Compute NIQE
                        try:
                            score = self._niqe_metric(str(frame_path)).item()
                            niqe_scores.append(score)
                        except Exception as e:
                            logger.debug(f"Failed to compute NIQE for frame {frame_idx}: {e}")

                        frame_idx += 1
            finally:
                cap.release()

            if not niqe_scores:
                return None

            return float(np.mean(niqe_scores))

        except Exception as e:
            logger.debug(f"NIQE video computation failed: {e}")
            return None

    def process(self, sample: Sample) -> Sample:
        """Process sample with NIQE metric."""
        if not self._ml_available:
            return sample

        try:
            # Compute NIQE score
            niqe_score = self.compute_nr_score(sample.path)

            if niqe_score is None:
                return sample

            # Store in quality metrics
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.niqe = niqe_score

            # Add validation issue if score is high (remember: lower is better for NIQE)
            if niqe_score > self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"High NIQE score: {niqe_score:.2f}",
                        details={"niqe": niqe_score, "threshold": self.warning_threshold},
                        recommendation="Image quality deviates from natural scene statistics. "
                        "May indicate distortions or artifacts.",
                    )
                )

            logger.debug(f"NIQE score for {sample.path.name}: {niqe_score:.2f}")

        except Exception as e:
            logger.warning(f"NIQE processing failed for {sample.path}: {e}")

        return sample
