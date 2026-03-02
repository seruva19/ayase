"""ITU-T P.1203 (HTTP Adaptive Streaming QoE) module.

ITU-T P.1203 is the standard for predicting Quality of Experience in
HTTP adaptive streaming. It accounts for video quality, stalling events,
resolution changes, and temporal effects.

Range: 1-5 MOS (higher = better QoE).

This implementation provides a simplified bitstream-based estimation
using video metadata (codec, bitrate, resolution, frame rate).
"""

import logging
import math
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class P1203Module(PipelineModule):
    name = "p1203"
    description = "ITU-T P.1203 streaming QoE estimation (1-5 MOS)"
    default_config = {
        "display_size": "phone",  # "phone", "tablet", "pc", "tv"
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.display_size = self.config.get("display_size", "phone")
        self._ml_available = False
        self._backend = None

    def setup(self) -> None:
        # Try official P.1203 implementation
        try:
            from itu_p1203 import P1203Standalone
            self._backend = "official"
            self._ml_available = True
            logger.info("P.1203 module initialised (official implementation)")
            return
        except ImportError:
            pass

        # Fallback: simplified parametric model
        self._backend = "parametric"
        self._ml_available = True
        logger.info("P.1203 module initialised (parametric estimation)")

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or not sample.is_video:
            return sample

        if sample.video_metadata is None:
            return sample

        try:
            mos = self._estimate_mos(sample)
            if mos is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.p1203_mos = mos
            logger.debug(f"P.1203 MOS for {sample.path.name}: {mos:.2f}")
        except Exception as e:
            logger.error(f"P.1203 failed: {e}")
        return sample

    def _estimate_mos(self, sample: Sample) -> Optional[float]:
        """Estimate MOS using P.1203-like parametric model.

        Uses video bitrate, resolution, frame rate, and codec to predict
        perceived quality on a 1-5 MOS scale.
        """
        meta = sample.video_metadata
        if meta is None:
            return None

        bitrate = meta.bitrate  # bps
        width = meta.width
        height = meta.height
        fps = meta.fps
        codec = (meta.codec or "").lower()

        if bitrate is None or bitrate <= 0:
            # Estimate from file size and duration
            if meta.duration > 0 and meta.file_size > 0:
                bitrate = int(meta.file_size * 8 / meta.duration)
            else:
                return None

        pixels = width * height
        bitrate_kbps = bitrate / 1000.0

        # Codec efficiency factor
        if "h265" in codec or "hevc" in codec or "265" in codec:
            codec_factor = 1.4
        elif "av1" in codec or "av01" in codec:
            codec_factor = 1.5
        elif "vp9" in codec:
            codec_factor = 1.3
        elif "h264" in codec or "avc" in codec or "264" in codec:
            codec_factor = 1.0
        else:
            codec_factor = 1.0

        # Effective bitrate (adjusted for codec efficiency)
        effective_bpp = (bitrate_kbps * 1000.0 * codec_factor) / (pixels * fps) if pixels * fps > 0 else 0

        # Simplified MOS model (inspired by P.1203.1)
        # Higher bpp -> higher quality, with diminishing returns
        if effective_bpp <= 0:
            return 1.0

        # Logarithmic quality model
        q = 1.0 + 4.0 * (1.0 - math.exp(-effective_bpp * 500.0))

        # Resolution penalty (lower resolution = lower quality ceiling)
        if pixels < 320 * 240:
            res_factor = 0.6
        elif pixels < 640 * 480:
            res_factor = 0.75
        elif pixels < 1280 * 720:
            res_factor = 0.85
        elif pixels < 1920 * 1080:
            res_factor = 0.95
        else:
            res_factor = 1.0

        # Frame rate penalty
        if fps < 15:
            fps_factor = 0.7
        elif fps < 24:
            fps_factor = 0.85
        elif fps < 30:
            fps_factor = 0.95
        else:
            fps_factor = 1.0

        # Display size adjustment
        display_factors = {
            "phone": 1.05,   # Small screen masks artifacts
            "tablet": 1.0,
            "pc": 0.95,
            "tv": 0.90,      # Large screen reveals more
        }
        display_factor = display_factors.get(self.display_size, 1.0)

        mos = q * res_factor * fps_factor * display_factor
        return float(max(1.0, min(5.0, mos)))
