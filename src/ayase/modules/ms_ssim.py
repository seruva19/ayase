"""MS-SSIM (Multi-Scale Structural Similarity Index) module.

MS-SSIM is an improvement over SSIM that computes similarity at multiple scales.
It correlates better with human perception than single-scale SSIM.
Range: 0-1 (higher is better). Typically 0.9+ is good quality.

This is a full-reference metric requiring a reference video for comparison.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ayase.models import Sample, ValidationIssue, ValidationSeverity, QualityMetrics
from ayase.base_modules import ReferenceBasedModule

logger = logging.getLogger(__name__)


class MSSSIMModule(ReferenceBasedModule):
    name = "ms_ssim"
    description = "Multi-Scale SSIM perceptual similarity metric (full-reference)"
    default_config = {
        "scales": 5,  # Number of downsampling scales
        "weights": [0.0448, 0.2856, 0.3001, 0.2363, 0.1333],  # Per-scale weights (Wang et al.)
        "subsample": 1,  # Process every Nth frame
        "warning_threshold": 0.85,  # Warn if MS-SSIM < 0.85
        "device": "auto",  # "cuda", "cpu", or "auto"
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.scales = self.config.get("scales", 5)
        self.weights = self.config.get(
            "weights", [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        )
        self.subsample = self.config.get("subsample", 1)
        self.warning_threshold = self.config.get("warning_threshold", 0.85)
        self.device_config = self.config.get("device", "auto")
        self.device = None
        self._ml_available = False
        self._ms_ssim_fn = None

    def setup(self) -> None:
        try:
            import torch
            from pytorch_msssim import ms_ssim

            self._ms_ssim_fn = ms_ssim

            # Set device
            if self.device_config == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(self.device_config)

            self._ml_available = True
            logger.info(f"MS-SSIM module initialized on {self.device}")

        except ImportError:
            logger.warning(
                "pytorch-msssim not installed. Install with: pip install pytorch-msssim"
            )
        except Exception as e:
            logger.warning(f"Failed to setup MS-SSIM: {e}")

    def compute_reference_score(
        self, sample_path: Path, reference_path: Path
    ) -> Optional[float]:
        """Compute MS-SSIM score comparing sample to reference.

        Args:
            sample_path: Path to distorted/test video
            reference_path: Path to reference/pristine video

        Returns:
            MS-SSIM score (0-1), or None if computation failed
        """
        try:
            # Open both videos
            ref_cap = cv2.VideoCapture(str(reference_path))
            dist_cap = cv2.VideoCapture(str(sample_path))

            ms_ssim_scores = []
            frame_idx = 0

            while True:
                ret_ref, ref_frame = ref_cap.read()
                ret_dist, dist_frame = dist_cap.read()

                if not ret_ref or not ret_dist:
                    break

                # Subsample frames
                if frame_idx % self.subsample != 0:
                    frame_idx += 1
                    continue

                # Convert BGR to RGB
                ref_rgb = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB)
                dist_rgb = cv2.cvtColor(dist_frame, cv2.COLOR_BGR2RGB)

                # Compute MS-SSIM for this frame
                try:
                    score = self._compute_ms_ssim_frame(ref_rgb, dist_rgb)
                    if score is not None:
                        ms_ssim_scores.append(score)
                except Exception as e:
                    logger.debug(f"Failed to compute MS-SSIM for frame {frame_idx}: {e}")

                frame_idx += 1

            ref_cap.release()
            dist_cap.release()

            if not ms_ssim_scores:
                return None

            return float(np.mean(ms_ssim_scores))

        except Exception as e:
            logger.warning(f"MS-SSIM computation failed: {e}")
            return None

    def _compute_ms_ssim_frame(self, ref_frame: np.ndarray, dist_frame: np.ndarray) -> Optional[float]:
        """Compute MS-SSIM for a single frame pair.

        Args:
            ref_frame: Reference frame (H, W, C) in RGB
            dist_frame: Distorted frame (H, W, C) in RGB

        Returns:
            MS-SSIM score (0-1), or None if computation failed
        """
        try:
            import torch

            # Convert to torch tensors (N, C, H, W)
            ref_tensor = (
                torch.from_numpy(ref_frame)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .to(self.device)
            )
            dist_tensor = (
                torch.from_numpy(dist_frame)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .to(self.device)
            )

            # Normalize to [0, 1] if needed (assuming input is [0, 255])
            if ref_tensor.max() > 1.0:
                ref_tensor = ref_tensor / 255.0
            if dist_tensor.max() > 1.0:
                dist_tensor = dist_tensor / 255.0

            # Compute MS-SSIM
            with torch.no_grad():
                ms_ssim_val = self._ms_ssim_fn(
                    ref_tensor,
                    dist_tensor,
                    data_range=1.0,
                    size_average=True,
                    win_size=11,
                )

            return ms_ssim_val.item()

        except Exception as e:
            logger.debug(f"MS-SSIM frame computation error: {e}")
            return None

    def process(self, sample: Sample) -> Sample:
        """Process sample with MS-SSIM metric.

        Checks for reference_path in sample metadata. If not found, skips processing.
        """
        if not self._ml_available:
            return sample

        if not sample.is_video:
            # For images, can still compute MS-SSIM
            return self._process_image(sample)

        # Check if sample has reference_path metadata
        reference = getattr(sample, "reference_path", None)
        if reference is None:
            return sample

        if not isinstance(reference, Path):
            reference = Path(reference)

        if not reference.exists():
            logger.debug(f"Reference video not found: {reference}")
            return sample

        try:
            # Compute MS-SSIM score
            ms_ssim_score = self.compute_reference_score(sample.path, reference)

            if ms_ssim_score is None:
                return sample

            # Store in quality metrics
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.ms_ssim = ms_ssim_score

            # Add validation issue if score is low
            if ms_ssim_score < self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low MS-SSIM score: {ms_ssim_score:.3f}",
                        details={
                            "ms_ssim": ms_ssim_score,
                            "threshold": self.warning_threshold,
                        },
                        recommendation="Structural similarity degraded compared to reference. "
                        "Check for distortions or quality loss.",
                    )
                )

            logger.debug(f"MS-SSIM score for {sample.path.name}: {ms_ssim_score:.3f}")

        except Exception as e:
            logger.warning(f"MS-SSIM processing failed for {sample.path}: {e}")

        return sample

    def _process_image(self, sample: Sample) -> Sample:
        """Process image sample with MS-SSIM.

        Args:
            sample: Image sample

        Returns:
            Sample with MS-SSIM score
        """
        reference = getattr(sample, "reference_path", None)
        if reference is None:
            return sample

        if not isinstance(reference, Path):
            reference = Path(reference)

        if not reference.exists():
            return sample

        try:
            # Load images
            ref_img = cv2.imread(str(reference))
            dist_img = cv2.imread(str(sample.path))

            if ref_img is None or dist_img is None:
                return sample

            # Convert BGR to RGB
            ref_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
            dist_rgb = cv2.cvtColor(dist_img, cv2.COLOR_BGR2RGB)

            # Compute MS-SSIM
            ms_ssim_score = self._compute_ms_ssim_frame(ref_rgb, dist_rgb)

            if ms_ssim_score is None:
                return sample

            # Store in quality metrics
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.ms_ssim = ms_ssim_score

            if ms_ssim_score < self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low MS-SSIM score: {ms_ssim_score:.3f}",
                        details={
                            "ms_ssim": ms_ssim_score,
                            "threshold": self.warning_threshold,
                        },
                    )
                )

        except Exception as e:
            logger.warning(f"MS-SSIM image processing failed for {sample.path}: {e}")

        return sample
