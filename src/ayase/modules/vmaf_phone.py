"""VMAF Phone module.

VMAF Phone is a variant of VMAF optimized for small-screen viewing
(mobile phones). It accounts for the fact that artifacts are less
visible on smaller screens.

Range: 0-100 (higher = better). Typically scores higher than standard
VMAF for the same content due to small-screen masking.

Full-reference metric. Requires FFmpeg with libvmaf.
"""

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.base_modules import ReferenceBasedModule

logger = logging.getLogger(__name__)


class VMAFPhoneModule(ReferenceBasedModule):
    name = "vmaf_phone"
    description = "VMAF phone model for mobile viewing (0-100, higher=better)"
    default_config = {}

    def __init__(self, config=None):
        super().__init__(config)
        self._ml_available = False

    def setup(self) -> None:
        try:
            result = subprocess.run(
                ["ffmpeg", "-filters"], capture_output=True, text=True, timeout=5
            )
            if "libvmaf" in result.stdout:
                self._ml_available = True
                logger.info("VMAF Phone module initialised")
        except Exception as e:
            logger.warning(f"VMAF Phone setup failed: {e}")

    def compute_reference_score(
        self, sample_path: Path, reference_path: Path
    ) -> Optional[float]:
        try:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                out = tmp.name
            cmd = [
                "ffmpeg", "-i", str(sample_path), "-i", str(reference_path),
                "-lavfi", f"[0:v][1:v]libvmaf=model=version=vmaf_v0.6.1:phone_model=1:log_path={out}:log_fmt=json",
                "-f", "null", "-",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                return None
            with open(out, "r") as f:
                data = json.load(f)
            Path(out).unlink(missing_ok=True)
            if "pooled_metrics" in data and "vmaf" in data["pooled_metrics"]:
                return float(data["pooled_metrics"]["vmaf"]["mean"])
            return None
        except Exception as e:
            logger.debug(f"VMAF Phone failed: {e}")
            return None

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or not sample.is_video:
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
            sample.quality_metrics.vmaf_phone = score
            logger.debug(f"VMAF Phone for {sample.path.name}: {score:.1f}")
        except Exception as e:
            logger.error(f"VMAF Phone failed: {e}")
        return sample
