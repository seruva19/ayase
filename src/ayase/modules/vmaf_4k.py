"""VMAF 4K module.

VMAF 4K uses a model optimized for 4K TV viewing at 1.5x screen height
distance. It captures sharpness differences at UHD resolutions that the
standard model may miss.

Range: 0-100 (higher = better).

Full-reference metric. Requires FFmpeg with libvmaf.
"""

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.base_modules import ReferenceBasedModule
from ayase.managed_ffmpeg import resolve_ffmpeg

logger = logging.getLogger(__name__)


class VMAF4KModule(ReferenceBasedModule):
    name = "vmaf_4k"
    implementation_status = "implemented"
    implementation_notes = "Managed ffmpeg libvmaf executes explicit vmaf_4k_v0.6.1 model on true reference/distorted video and parses pooled vmaf mean."
    implementation_review = "Source inspection; confidence high; pretrained inference not validated."
    implementation_sources = ("../managed_ffmpeg.py",)
    provenance = "published"
    sources = {
        "vmaf_4k": "VMAF 4K v0.6.1; libvmaf — https://github.com/Netflix/vmaf",
    }
    description = "VMAF 4K model for UHD content (0-100, higher=better)"
    default_config = {"ffmpeg_path": None}
    metric_groups = {
        "vmaf_4k": "fr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._ml_available = False
        self._backend = None
        self._ffmpeg_path: Optional[Path] = None

    def setup(self) -> None:
        if self.test_mode:
            return
        try:
            self._ffmpeg_path = resolve_ffmpeg(self.config.get("ffmpeg_path"))
            self._ml_available = True
            self._backend = "ffmpeg_libvmaf"
            logger.info("VMAF 4K module initialised")
        except Exception as e:
            self._backend = "unavailable"
            logger.warning(f"VMAF 4K setup failed: {e}")

    def compute_reference_score(
        self, sample_path: Path, reference_path: Path
    ) -> Optional[float]:
        if self._ffmpeg_path is None:
            return None
        try:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                out = tmp.name
            cmd = [
                str(self._ffmpeg_path), "-i", str(sample_path), "-i", str(reference_path),
                "-lavfi", f"[0:v][1:v]libvmaf=model=version=vmaf_4k_v0.6.1:log_path={out}:log_fmt=json",
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
            logger.debug(f"VMAF 4K failed: {e}")
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
            sample.quality_metrics.vmaf_4k = score
            logger.debug(f"VMAF 4K for {sample.path.name}: {score:.1f}")
        except Exception as e:
            logger.error(f"VMAF 4K failed: {e}")
        return sample
