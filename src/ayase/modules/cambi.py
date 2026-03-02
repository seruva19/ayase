"""CAMBI (Contrast Aware Multiscale Banding Index) module.

CAMBI is Netflix's no-reference banding/contouring artifact detector.
It detects false contours (banding) caused by insufficient bit depth
or aggressive quantization in video encoding.

Range: 0-24 (lower = less banding, 0 = no banding detected).
Values above 5 typically indicate visible banding.

Requires FFmpeg with libvmaf support (CAMBI is built into libvmaf).
"""

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from ayase.models import Sample, QualityMetrics, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class CAMBIModule(PipelineModule):
    name = "cambi"
    description = "CAMBI banding/contouring detector (Netflix, 0-24, lower=better)"
    default_config = {
        "warning_threshold": 5.0,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.warning_threshold = self.config.get("warning_threshold", 5.0)
        self._ml_available = False

    def setup(self) -> None:
        try:
            result = subprocess.run(
                ["ffmpeg", "-filters"], capture_output=True, text=True, timeout=5
            )
            if "libvmaf" in result.stdout:
                self._ml_available = True
                logger.info("CAMBI module initialised (FFmpeg libvmaf)")
            else:
                logger.warning("FFmpeg libvmaf not available for CAMBI")
        except FileNotFoundError:
            logger.warning("FFmpeg not found. CAMBI requires FFmpeg with libvmaf.")
        except Exception as e:
            logger.warning(f"Failed to setup CAMBI: {e}")

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or not sample.is_video:
            return sample

        try:
            score = self._compute_cambi(sample.path)
            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.cambi = score

            if score > self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Banding detected (CAMBI={score:.1f})",
                        details={"cambi": score, "threshold": self.warning_threshold},
                        recommendation="Visible banding/contouring. Consider higher bit depth or less aggressive quantization.",
                    )
                )
            logger.debug(f"CAMBI for {sample.path.name}: {score:.2f}")
        except Exception as e:
            logger.error(f"CAMBI failed for {sample.path}: {e}")
        return sample

    def _compute_cambi(self, video_path: Path) -> Optional[float]:
        try:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                output_path = tmp.name

            cmd = [
                "ffmpeg", "-i", str(video_path),
                "-lavfi", f"libvmaf=feature=name=cambi:log_path={output_path}:log_fmt=json",
                "-f", "null", "-",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                logger.debug(f"CAMBI FFmpeg failed: {result.stderr[:200]}")
                return None

            with open(output_path, "r") as f:
                data = json.load(f)

            Path(output_path).unlink(missing_ok=True)

            if "pooled_metrics" in data and "cambi" in data["pooled_metrics"]:
                return float(data["pooled_metrics"]["cambi"]["mean"])
            elif "frames" in data:
                scores = [
                    fr["metrics"].get("cambi", 0) for fr in data["frames"]
                    if "cambi" in fr.get("metrics", {})
                ]
                return float(np.mean(scores)) if scores else None
            return None
        except subprocess.TimeoutExpired:
            logger.warning(f"CAMBI timed out for {video_path}")
            return None
        except Exception as e:
            logger.debug(f"CAMBI computation failed: {e}")
            return None
