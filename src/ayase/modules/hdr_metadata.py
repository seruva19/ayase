"""MaxCLL/MaxFALL per CTA-861.3, measured on true HDR code values.

Per CTA-861.3 both statistics are computed on the per-pixel maximum of the
linear-light R/G/B channels expressed in cd/m². Frames are decoded through
FFmpeg as 16-bit-per-channel RGB (``rgb48le``), which preserves the PQ signal
encoding; the ST.2084 EOTF then maps code values to absolute luminance.

``max_cll`` — largest pixel value of max(R,G,B) across the video, in nits.
``max_fall`` — largest frame-average of max(R,G,B), in nits.

Only PQ (smpte2084) content produces values; SDR and HLG inputs are skipped —
MaxCLL/MaxFALL are not defined for them without display assumptions.
"""

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


def _pq_eotf(signal: np.ndarray) -> np.ndarray:
    """Apply ST.2084 PQ EOTF (signal → linear light, 0-10000 nits).

    Converts PQ-encoded signal values [0,1] to absolute luminance in nits.
    """
    signal = np.clip(signal, 0.0, 1.0)
    m1 = 0.1593017578125
    m2 = 78.84375
    c1 = 0.8359375
    c2 = 18.8515625
    c3 = 18.6875

    Vm2 = np.power(signal, 1.0 / m2)
    num = np.maximum(Vm2 - c1, 0.0)
    den = c2 - c3 * Vm2
    den = np.maximum(den, 1e-10)
    linear = np.power(num / den, 1.0 / m1)
    return linear * 10000.0  # nits


def _color_transfer(path: Path) -> Optional[str]:
    """Probe the video's color transfer characteristic via ffprobe."""
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=color_transfer,pix_fmt",
                "-of", "json", str(path),
            ],
            capture_output=True, text=True, timeout=30,
        )
        streams = json.loads(out.stdout).get("streams") or []
        if streams:
            return streams[0].get("color_transfer")
    except Exception as e:
        logger.debug("ffprobe color_transfer failed for %s: %s", path, e)
    return None


class HDRMetadataModule(PipelineModule):
    name = "hdr_metadata"
    provenance = "published"
    sources = {
        "max_cll": "MaxCLL/MaxFALL (CTA-861.3; https://shop.cta.tech/products/cta-861-3; ST.2084 PQ decode via FFmpeg)",
        "max_fall": "MaxCLL/MaxFALL (CTA-861.3; https://shop.cta.tech/products/cta-861-3; ST.2084 PQ decode via FFmpeg)",
    }
    deviations = {
        "max_cll": "values only for PQ content (color_transfer=smpte2084); SDR/HLG are skipped — CTA-861.3 is undefined for them without display assumptions",
        "max_fall": "values only for PQ content (color_transfer=smpte2084); SDR/HLG are skipped — CTA-861.3 is undefined for them without display assumptions",
    }
    description = "MaxFALL + MaxCLL per CTA-861.3 (PQ nits)"
    default_config = {}
    metric_groups = {
        "max_cll": "hdr",
        "max_fall": "hdr",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._ml_available = shutil.which("ffmpeg") is not None
        self._backend = "algorithmic"

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video or not self._ml_available:
            return sample

        try:
            transfer = _color_transfer(sample.path)
            if transfer != "smpte2084":
                logger.debug(
                    "hdr_metadata: %s transfer=%r — not PQ, skipping",
                    sample.path.name, transfer,
                )
                return sample

            max_fall, max_cll = self._analyze_video(sample.path)
            if max_fall is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.max_fall = max_fall
            sample.quality_metrics.max_cll = max_cll
            logger.debug(f"HDR metadata for {sample.path.name}: MaxFALL={max_fall:.1f} MaxCLL={max_cll:.1f} nits")
        except Exception as e:
            logger.error(f"HDR metadata failed: {e}")
        return sample

    def _analyze_video(self, path: Path):
        """Stream every frame as rgb48le and accumulate CTA-861.3 statistics."""
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height", "-of", "json", str(path),
            ],
            capture_output=True, text=True, timeout=30,
        )
        streams = json.loads(probe.stdout).get("streams") or []
        if not streams:
            return None, None
        width = int(streams[0]["width"])
        height = int(streams[0]["height"])
        frame_bytes = width * height * 3 * 2  # rgb48le

        proc = subprocess.Popen(
            [
                "ffmpeg", "-v", "error", "-i", str(path),
                "-f", "rawvideo", "-pix_fmt", "rgb48le", "-",
            ],
            stdout=subprocess.PIPE,
        )
        frame_averages = []
        pixel_max = 0.0
        try:
            while True:
                buf = proc.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    break
                rgb = np.frombuffer(buf, dtype="<u2").reshape(height, width, 3)
                # CTA-861.3: per-pixel max of the three channels, then EOTF.
                signal = rgb.max(axis=2).astype(np.float64) / 65535.0
                lum_nits = _pq_eotf(signal)
                frame_averages.append(float(lum_nits.mean()))
                pixel_max = max(pixel_max, float(lum_nits.max()))
        finally:
            proc.stdout.close()
            proc.wait()

        if not frame_averages:
            return None, None
        return float(max(frame_averages)), float(pixel_max)
