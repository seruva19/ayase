"""VMAF (Video Multimethod Assessment Fusion) module.

VMAF is an industry-standard perceptual video quality metric developed by Netflix.
It combines multiple quality models to predict human perception of video quality.
Range: 0-100 (higher is better). Typically 80+ is excellent quality.

This is a full-reference metric requiring a reference video for comparison.
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
import json

import cv2
import numpy as np

from ayase.models import Sample, ValidationIssue, ValidationSeverity, QualityMetrics
from ayase.base_modules import ReferenceBasedModule

logger = logging.getLogger(__name__)


class VMAFModule(ReferenceBasedModule):
    name = "vmaf"
    description = "VMAF perceptual video quality metric (full-reference)"
    default_config = {
        "vmaf_model": "vmaf_v0.6.1",  # or "vmaf_4k_v0.6.1" for 4K content
        "subsample": 1,  # Process every Nth frame (1=all frames)
        "use_ffmpeg": True,  # Use FFmpeg libvmaf if available, else fallback to frame-by-frame
        "warning_threshold": 70.0,  # Warn if VMAF < 70
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.vmaf_model = self.config.get("vmaf_model", "vmaf_v0.6.1")
        self.subsample = self.config.get("subsample", 1)
        self.use_ffmpeg = self.config.get("use_ffmpeg", True)
        self.warning_threshold = self.config.get("warning_threshold", 70.0)
        self._ml_available = False
        self._ffmpeg_vmaf_available = False

    def setup(self) -> None:
        try:
            # Check if FFmpeg with libvmaf is available
            if self.use_ffmpeg:
                result = subprocess.run(
                    ["ffmpeg", "-filters"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if "libvmaf" in result.stdout or "vmaf" in result.stdout:
                    self._ffmpeg_vmaf_available = True
                    logger.info("FFmpeg with libvmaf support detected")
                else:
                    logger.warning(
                        "FFmpeg found but libvmaf not available. "
                        "Falling back to Python implementation."
                    )

            # Try to import vmaf package as fallback
            try:
                import vmaf

                self._ml_available = True
                logger.info(f"VMAF module initialized (model: {self.vmaf_model})")
            except ImportError:
                if not self._ffmpeg_vmaf_available:
                    logger.warning(
                        "VMAF package not installed and FFmpeg libvmaf not available. "
                        "Install with: pip install vmaf"
                    )
                else:
                    self._ml_available = True  # FFmpeg method available

        except Exception as e:
            logger.warning(f"Failed to setup VMAF: {e}")

    def compute_reference_score(
        self, sample_path: Path, reference_path: Path
    ) -> Optional[float]:
        """Compute VMAF score comparing sample to reference.

        Args:
            sample_path: Path to distorted/test video
            reference_path: Path to reference/pristine video

        Returns:
            VMAF score (0-100), or None if computation failed
        """
        if self._ffmpeg_vmaf_available:
            return self._compute_vmaf_ffmpeg(sample_path, reference_path)
        else:
            return self._compute_vmaf_python(sample_path, reference_path)

    def _compute_vmaf_ffmpeg(
        self, sample_path: Path, reference_path: Path
    ) -> Optional[float]:
        """Compute VMAF using FFmpeg's libvmaf filter.

        This is the fastest and most accurate method.
        """
        try:
            # Create temporary file for VMAF JSON output
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as tmp_file:
                output_path = tmp_file.name

            # Build FFmpeg command
            # FFmpeg VMAF filter: -lavfi "[0:v][1:v]libvmaf=model=version={model}:log_path={output}"
            model_version = self.vmaf_model.replace("vmaf_", "").replace("_", ".")

            cmd = [
                "ffmpeg",
                "-i",
                str(sample_path),  # Distorted video (input 0)
                "-i",
                str(reference_path),  # Reference video (input 1)
                "-lavfi",
                f"[0:v][1:v]libvmaf=model=version={model_version}:log_path={output_path}:log_fmt=json",
                "-f",
                "null",
                "-",
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300  # 5 min timeout
            )

            if result.returncode != 0:
                logger.warning(f"FFmpeg VMAF failed: {result.stderr}")
                return None

            # Parse VMAF JSON output
            with open(output_path, "r") as f:
                vmaf_data = json.load(f)

            # Extract mean VMAF score
            if "pooled_metrics" in vmaf_data:
                vmaf_score = vmaf_data["pooled_metrics"]["vmaf"]["mean"]
            elif "frames" in vmaf_data:
                # Average frame scores
                frame_scores = [
                    frame["metrics"]["vmaf"] for frame in vmaf_data["frames"]
                ]
                vmaf_score = np.mean(frame_scores)
            else:
                logger.warning("Unexpected VMAF JSON format")
                return None

            # Clean up temp file
            Path(output_path).unlink(missing_ok=True)

            return float(vmaf_score)

        except subprocess.TimeoutExpired:
            logger.warning(f"VMAF computation timed out for {sample_path}")
            return None
        except Exception as e:
            logger.warning(f"FFmpeg VMAF failed: {e}")
            return None

    def _compute_vmaf_python(
        self, sample_path: Path, reference_path: Path
    ) -> Optional[float]:
        """Compute VMAF using Python vmaf package (frame-by-frame).

        Fallback method if FFmpeg libvmaf not available.
        """
        try:
            import vmaf

            # Open both videos
            ref_cap = cv2.VideoCapture(str(reference_path))
            dist_cap = cv2.VideoCapture(str(sample_path))

            vmaf_scores = []
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

                # Compute VMAF for this frame
                # Note: vmaf package API may vary, this is a simplified example
                try:
                    score = vmaf.compute_vmaf(
                        ref_rgb, dist_rgb, model=self.vmaf_model
                    )
                    vmaf_scores.append(score)
                except Exception as e:
                    logger.debug(f"Failed to compute VMAF for frame {frame_idx}: {e}")

                frame_idx += 1

            ref_cap.release()
            dist_cap.release()

            if not vmaf_scores:
                return None

            return float(np.mean(vmaf_scores))

        except ImportError:
            logger.warning("vmaf package not installed")
            return None
        except Exception as e:
            logger.warning(f"Python VMAF computation failed: {e}")
            return None

    def process(self, sample: Sample) -> Sample:
        """Process sample with VMAF metric.

        Checks for reference_path in sample metadata. If not found, skips processing.
        """
        if not self._ml_available and not self._ffmpeg_vmaf_available:
            return sample

        if not sample.is_video:
            return sample  # VMAF is for videos only

        # Check if sample has reference_path metadata
        reference = getattr(sample, "reference_path", None)
        if reference is None:
            # No reference available, skip gracefully
            return sample

        if not isinstance(reference, Path):
            reference = Path(reference)

        if not reference.exists():
            logger.debug(f"Reference video not found: {reference}")
            return sample

        try:
            # Compute VMAF score
            vmaf_score = self.compute_reference_score(sample.path, reference)

            if vmaf_score is None:
                return sample

            # Store in quality metrics
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.vmaf = vmaf_score

            # Add validation issue if score is low
            if vmaf_score < self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low VMAF score: {vmaf_score:.1f}",
                        details={"vmaf": vmaf_score, "threshold": self.warning_threshold},
                        recommendation="Video quality degraded compared to reference. "
                        "Check for compression artifacts or encoding issues.",
                    )
                )

            logger.debug(f"VMAF score for {sample.path.name}: {vmaf_score:.1f}")

        except Exception as e:
            logger.warning(f"VMAF processing failed for {sample.path}: {e}")

        return sample
