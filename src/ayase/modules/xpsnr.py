"""XPSNR (Extended Perceptually Weighted PSNR) module.

XPSNR is a psychovisually motivated distortion metric developed by
Fraunhofer HHI. It is integrated into FFmpeg and provides PSNR values
weighted by the human visual system's sensitivity.

Range: dB scale (higher = better, typically 25-50 dB).

This is a full-reference metric. Requires FFmpeg with xpsnr filter.
"""

import logging
import re
import subprocess
from pathlib import Path
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.base_modules import ReferenceBasedModule

logger = logging.getLogger(__name__)


class XPSNRModule(ReferenceBasedModule):
    name = "xpsnr"
    description = "XPSNR perceptually weighted PSNR (Fraunhofer, dB, higher=better)"
    default_config = {}

    def __init__(self, config=None):
        super().__init__(config)
        self._ml_available = False

    def setup(self) -> None:
        try:
            result = subprocess.run(
                ["ffmpeg", "-filters"], capture_output=True, text=True, timeout=5
            )
            if "xpsnr" in result.stdout:
                self._ml_available = True
                logger.info("XPSNR module initialised (FFmpeg xpsnr filter)")
            else:
                logger.warning("FFmpeg xpsnr filter not available")
        except FileNotFoundError:
            logger.warning("FFmpeg not found")
        except Exception as e:
            logger.warning(f"Failed to setup XPSNR: {e}")

    def compute_reference_score(
        self, sample_path: Path, reference_path: Path
    ) -> Optional[float]:
        try:
            cmd = [
                "ffmpeg",
                "-i", str(sample_path),
                "-i", str(reference_path),
                "-lavfi", "[0:v][1:v]xpsnr",
                "-f", "null", "-",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            # Parse XPSNR from stderr (FFmpeg outputs stats there)
            output = result.stderr
            match = re.search(r"XPSNR\s+[Aa]verage[:\s]+([0-9.]+)", output)
            if match:
                return float(match.group(1))

            # Try alternative pattern
            match = re.search(r"XPSNR\s*y:\s*([0-9.]+)", output)
            if match:
                return float(match.group(1))

            logger.debug("Could not parse XPSNR from output")
            return None
        except subprocess.TimeoutExpired:
            return None
        except Exception as e:
            logger.debug(f"XPSNR failed: {e}")
            return None

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
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
            sample.quality_metrics.xpsnr = score
            logger.debug(f"XPSNR for {sample.path.name}: {score:.2f} dB")
        except Exception as e:
            logger.error(f"XPSNR failed for {sample.path}: {e}")
        return sample
