"""VIF (Visual Information Fidelity) module.

VIF is a full-reference image quality metric based on natural scene statistics
and information theory. It measures the information shared between reference
and distorted images.
Range: 0-1 (higher is better). Typical good quality: 0.4+
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ayase.models import Sample, ValidationIssue, ValidationSeverity, QualityMetrics
from ayase.base_modules import ReferenceBasedModule

logger = logging.getLogger(__name__)


class VIFModule(ReferenceBasedModule):
    name = "vif"
    description = "Visual Information Fidelity metric (full-reference)"
    default_config = {
        "subsample": 1,  # Process every Nth frame
        "warning_threshold": 0.3,  # Warn if VIF < 0.3
        "device": "auto",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 1)
        self.warning_threshold = self.config.get("warning_threshold", 0.3)
        self.device_config = self.config.get("device", "auto")
        self.device = None
        self._ml_available = False
        self._vif_fn = None

    def setup(self) -> None:
        try:
            import torch
            import piq

            self._vif_fn = piq.vif_p  # Pixel-domain VIF

            # Set device
            if self.device_config == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(self.device_config)

            self._ml_available = True
            logger.info(f"VIF module initialized on {self.device}")

        except ImportError:
            logger.warning("piq package not installed. Install with: pip install piq")
        except Exception as e:
            logger.warning(f"Failed to setup VIF: {e}")

    def compute_reference_score(
        self, sample_path: Path, reference_path: Path
    ) -> Optional[float]:
        """Compute VIF score comparing sample to reference.

        Args:
            sample_path: Path to distorted/test video/image
            reference_path: Path to reference/pristine video/image

        Returns:
            VIF score (0-1), or None if computation failed
        """
        try:
            # Check if video or image
            sample_str = str(sample_path)
            if sample_str.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                return self._compute_vif_video(sample_path, reference_path)
            else:
                return self._compute_vif_image(sample_path, reference_path)

        except Exception as e:
            logger.warning(f"VIF computation failed: {e}")
            return None

    def _compute_vif_image(
        self, sample_path: Path, reference_path: Path
    ) -> Optional[float]:
        """Compute VIF for a single image pair."""
        try:
            import torch

            # Load images
            ref_img = cv2.imread(str(reference_path))
            dist_img = cv2.imread(str(sample_path))

            if ref_img is None or dist_img is None:
                return None

            # Convert BGR to RGB
            ref_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
            dist_rgb = cv2.cvtColor(dist_img, cv2.COLOR_BGR2RGB)

            # Convert to torch tensors (N, C, H, W)
            ref_tensor = (
                torch.from_numpy(ref_rgb)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .to(self.device)
            )
            dist_tensor = (
                torch.from_numpy(dist_rgb)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .to(self.device)
            )

            # Normalize to [0, 1]
            if ref_tensor.max() > 1.0:
                ref_tensor = ref_tensor / 255.0
            if dist_tensor.max() > 1.0:
                dist_tensor = dist_tensor / 255.0

            # Compute VIF
            with torch.no_grad():
                vif_score = self._vif_fn(dist_tensor, ref_tensor, data_range=1.0)

            return vif_score.item()

        except Exception as e:
            logger.debug(f"VIF image computation failed: {e}")
            return None

    def _compute_vif_video(
        self, sample_path: Path, reference_path: Path
    ) -> Optional[float]:
        """Compute VIF for video (average across frames)."""
        try:
            import torch

            ref_cap = cv2.VideoCapture(str(reference_path))
            dist_cap = cv2.VideoCapture(str(sample_path))

            vif_scores = []
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

                # Convert to tensors
                ref_tensor = (
                    torch.from_numpy(ref_rgb)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .float()
                    .to(self.device)
                    / 255.0
                )
                dist_tensor = (
                    torch.from_numpy(dist_rgb)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .float()
                    .to(self.device)
                    / 255.0
                )

                # Compute VIF
                with torch.no_grad():
                    vif_score = self._vif_fn(dist_tensor, ref_tensor, data_range=1.0)
                    vif_scores.append(vif_score.item())

                frame_idx += 1

            ref_cap.release()
            dist_cap.release()

            if not vif_scores:
                return None

            return float(np.mean(vif_scores))

        except Exception as e:
            logger.debug(f"VIF video computation failed: {e}")
            return None

    def process(self, sample: Sample) -> Sample:
        """Process sample with VIF metric."""
        if not self._ml_available:
            return sample

        # Check if sample has reference_path metadata
        reference = getattr(sample, "reference_path", None)
        if reference is None:
            return sample

        if not isinstance(reference, Path):
            reference = Path(reference)

        if not reference.exists():
            return sample

        try:
            # Compute VIF score
            vif_score = self.compute_reference_score(sample.path, reference)

            if vif_score is None:
                return sample

            # Store in quality metrics
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.vif = vif_score

            # Add validation issue if score is low
            if vif_score < self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low VIF score: {vif_score:.3f}",
                        details={"vif": vif_score, "threshold": self.warning_threshold},
                        recommendation="Visual information fidelity is low. "
                        "Significant information loss compared to reference.",
                    )
                )

            logger.debug(f"VIF score for {sample.path.name}: {vif_score:.3f}")

        except Exception as e:
            logger.warning(f"VIF processing failed for {sample.path}: {e}")

        return sample
