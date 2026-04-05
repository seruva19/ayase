"""Variable Frame Rate (VFR) and temporal jitter detection via ffprobe.

Analyzes frame timestamp consistency to detect VFR containers that may cause
inconsistent motion dynamics. Reports max and average jitter in milliseconds."""

import logging
import subprocess
import json
import shutil
import numpy as np
from typing import Optional

from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

class VFRDetectionModule(PipelineModule):
    """
    Detects Variable Frame Rate (VFR) jitter by analyzing frame timestamps.
    VFR can cause inconsistent motion dynamics training.
    """
    name = "vfr_detection"
    description = "Variable Frame Rate (VFR) and jitter detection"
    default_config = {
        "jitter_threshold_ms": 2.0,  # Max allowed variance in frame duration
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.jitter_threshold_ms = self.config.get("jitter_threshold_ms", 2.0)
        self._ffprobe_available = False

    def setup(self) -> None:
        self._ffprobe_available = shutil.which("ffprobe") is not None
        if not self._ffprobe_available:
            logger.warning("ffprobe not found. VFRDetectionModule disabled.")

    def process(self, sample: Sample) -> Sample:
        if not self._ffprobe_available or not sample.is_video:
            return sample

        try:
            # ffprobe -show_frames -select_streams v -print_format json 
            # This is slow for long videos. We sample the first 100 frames.
            cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "frame=best_effort_timestamp_time",
                "-read_intervals", "%+10", # First 10 seconds of data
                "-print_format", "json",
                str(sample.path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return sample

            data = json.loads(result.stdout)
            frames = data.get("frames", [])
            
            if len(frames) < 10:
                return sample

            timestamps = [float(f["best_effort_timestamp_time"]) for f in frames if "best_effort_timestamp_time" in f]
            if len(timestamps) < 2:
                return sample

            # Calculate durations between frames
            durations = np.diff(timestamps)
            
            # Use median as the target "constant" duration
            median_duration = np.median(durations)
            
            # Calculate jitter (variance from median)
            jitter = np.abs(durations - median_duration) * 1000.0 # to ms
            max_jitter = np.max(jitter)
            avg_jitter = np.mean(jitter)

            if max_jitter > self.jitter_threshold_ms:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Variable Frame Rate (VFR) detected. Jitter: {max_jitter:.2f}ms",
                        details={
                            "max_jitter_ms": float(max_jitter),
                            "avg_jitter_ms": float(avg_jitter),
                            "median_fps": 1.0 / median_duration if median_duration > 0 else 0
                        },
                        recommendation="Significant temporal jitter detected. Resample to Constant Frame Rate (CFR) to ensure physical consistency in learned motion."
                    )
                )

        except Exception as e:
            logger.warning(f"VFR detection failed for {sample.path}: {e}")

        return sample
