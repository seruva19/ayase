"""VMAF NEG (No Enhancement Gain) module.

VMAF NEG is a variant of VMAF developed by Netflix that penalises
artificial enhancement / sharpening. Standard VMAF can be "gamed" by
applying sharpening filters; VMAF NEG addresses this by using a
negative-gain-aware model.

Range: 0-100 (higher = better, same as VMAF).

This is a full-reference metric. Requires FFmpeg with libvmaf support.
"""

import logging
import subprocess
import tempfile
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics, ValidationIssue, ValidationSeverity
from ayase.base_modules import ReferenceBasedModule

logger = logging.getLogger(__name__)


class VMAFNEGModule(ReferenceBasedModule):
    name = "vmaf_neg"
    description = "VMAF NEG no-enhancement-gain variant (0-100, higher=better)"
    default_config = {
        "subsample": 1,
        "warning_threshold": 70.0,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 1)
        self.warning_threshold = self.config.get("warning_threshold", 70.0)
        self._ml_available = False
        self._ffmpeg_available = False

    def setup(self) -> None:
        try:
            result = subprocess.run(
                ["ffmpeg", "-filters"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if "libvmaf" in result.stdout or "vmaf" in result.stdout:
                self._ffmpeg_available = True
                self._ml_available = True
                logger.info("VMAF NEG module initialised (FFmpeg libvmaf)")
            else:
                logger.warning(
                    "FFmpeg found but libvmaf not available. "
                    "VMAF NEG requires FFmpeg with libvmaf support."
                )
        except FileNotFoundError:
            logger.warning("FFmpeg not found. VMAF NEG requires FFmpeg with libvmaf.")
        except Exception as e:
            logger.warning(f"Failed to setup VMAF NEG: {e}")

    def compute_reference_score(
        self, sample_path: Path, reference_path: Path
    ) -> Optional[float]:
        if not self._ffmpeg_available:
            return None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as tmp_file:
                output_path = tmp_file.name

            # Use vmaf_neg model via libvmaf
            # The NEG model is available as "vmaf_v0.6.1neg" or via neg=true flag
            cmd = [
                "ffmpeg",
                "-i", str(sample_path),
                "-i", str(reference_path),
                "-lavfi",
                f"[0:v][1:v]libvmaf=model=version=vmaf_v0.6.1neg:log_path={output_path}:log_fmt=json",
                "-f", "null",
                "-",
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )

            if result.returncode != 0:
                # Try alternative: model=path with neg model
                cmd_alt = [
                    "ffmpeg",
                    "-i", str(sample_path),
                    "-i", str(reference_path),
                    "-lavfi",
                    f"[0:v][1:v]libvmaf=model=version=vmaf_v0.6.1:neg=true:log_path={output_path}:log_fmt=json",
                    "-f", "null",
                    "-",
                ]
                result = subprocess.run(
                    cmd_alt, capture_output=True, text=True, timeout=300
                )
                if result.returncode != 0:
                    logger.warning(f"VMAF NEG FFmpeg failed: {result.stderr[:200]}")
                    return None

            with open(output_path, "r") as f:
                vmaf_data = json.load(f)

            if "pooled_metrics" in vmaf_data:
                # Look for vmaf_neg or vmaf key
                metrics = vmaf_data["pooled_metrics"]
                if "vmaf_neg" in metrics:
                    score = metrics["vmaf_neg"]["mean"]
                elif "vmaf" in metrics:
                    score = metrics["vmaf"]["mean"]
                else:
                    score = list(metrics.values())[0]["mean"]
            elif "frames" in vmaf_data:
                frame_scores = []
                for frame in vmaf_data["frames"]:
                    m = frame.get("metrics", {})
                    s = m.get("vmaf_neg", m.get("vmaf", None))
                    if s is not None:
                        frame_scores.append(s)
                if not frame_scores:
                    return None
                score = np.mean(frame_scores)
            else:
                return None

            Path(output_path).unlink(missing_ok=True)
            return float(score)

        except subprocess.TimeoutExpired:
            logger.warning(f"VMAF NEG timed out for {sample_path}")
            return None
        except Exception as e:
            logger.warning(f"VMAF NEG failed: {e}")
            return None

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        if not sample.is_video:
            return sample

        reference = getattr(sample, "reference_path", None)
        if reference is None:
            return sample
        reference = Path(reference) if not isinstance(reference, Path) else reference
        if not reference.exists():
            return sample

        try:
            score = self.compute_reference_score(sample.path, reference)
            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.vmaf_neg = score

            if score < self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low VMAF NEG score: {score:.1f}",
                        details={"vmaf_neg": score, "threshold": self.warning_threshold},
                        recommendation=(
                            "Video quality degraded (NEG variant penalises "
                            "artificial sharpening/enhancement)."
                        ),
                    )
                )
            logger.debug(f"VMAF NEG for {sample.path.name}: {score:.1f}")
        except Exception as e:
            logger.warning(f"VMAF NEG processing failed for {sample.path}: {e}")
        return sample
