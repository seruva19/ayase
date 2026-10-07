"""ITU-T P.1203 (HTTP Adaptive Streaming QoE) module.

ITU-T P.1203 is the standard for predicting Quality of Experience in
HTTP adaptive streaming. It accounts for video quality, stalling events,
resolution changes, and temporal effects.

Range: 1-5 MOS (higher = better QoE).

This module uses the upstream ITU-T P.1203 implementation (the ``itu_p1203``
package), building the per-segment description from the decoded video's
metadata. When ``itu_p1203`` is not installed the module reports no score
rather than substituting an ad-hoc parametric approximation.
"""

import logging
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class P1203Module(PipelineModule):
    name = "p1203"
    provenance = "published"
    sources = {
        "p1203_mos": "ITU-T P.1203 via itu-p1203 — https://github.com/itu-p1203/itu-p1203",
    }
    deviations = {
        "p1203_mos": "Input is a synthesized single-segment mode-0 report built "
        "from container metadata (no per-segment representations, no stalling "
        "or audio data) — h264 only.",
    }
    description = "ITU-T P.1203 streaming QoE estimation (1-5 MOS)"
    default_config = {
        "display_size": "phone",  # "phone", "tablet", "pc", "tv"
    }
    metric_groups = {
        "p1203_mos": "audio",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.display_size = self.config.get("display_size", "phone")
        self._ml_available = False
        self._backend = None

    def setup(self) -> None:
        # Only the upstream P.1203 implementation is a real backend.
        try:
            from itu_p1203 import P1203Standalone
            self._p1203_cls = P1203Standalone
            self._backend = "itu_p1203"
            self._ml_available = True
            logger.info("P.1203 module initialised (upstream implementation)")
        except ImportError:
            self._p1203_cls = None
            self._backend = "unavailable"
            self._ml_available = False
            logger.warning(
                "P.1203 unavailable: upstream itu_p1203 not installed "
                "(pip install itu-p1203); no score emitted."
            )

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or not sample.is_video:
            return sample

        if sample.video_metadata is None:
            return sample

        try:
            mos = self._compute_official(sample)
            if mos is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.p1203_mos = mos
            logger.debug(f"P.1203 MOS for {sample.path.name}: {mos:.2f}")
        except Exception as e:
            logger.error(f"P.1203 failed: {e}")
        return sample

    def _compute_official(self, sample: Sample) -> Optional[float]:
        """Compute MOS using the upstream ITU-T P.1203 implementation.

        Builds the JSON input dict the official ``itu_p1203`` package
        expects: ``I13`` video segments, ``I23`` stalling info, ``IGen``
        generic display/device info. Mode 0 supports only h264 — other
        codecs need the P.1204 extension, so we decline rather than
        fabricate a score.
        """
        meta = sample.video_metadata
        if meta is None:
            return None

        try:
            codec = (meta.codec or "h264").lower()
            if "h264" not in codec and "avc" not in codec:
                logger.debug("P.1203 mode 0 supports only h264; codec=%s", codec)
                return None

            bitrate = meta.bitrate
            if bitrate is None or bitrate <= 0:
                if meta.duration > 0 and meta.file_size and meta.file_size > 0:
                    bitrate = int(meta.file_size * 8 / meta.duration)
                else:
                    return None

            device_map = {
                "phone": "mobile",
                "tablet": "handheld",
                "pc": "pc",
                "tv": "pc",
            }
            input_json = {
                "I13": {
                    "streamId": 42,
                    "segments": [
                        {
                            "codec": "h264",
                            "start": 0.0,
                            "duration": float(meta.duration),
                            "resolution": f"{meta.width}x{meta.height}",
                            "bitrate": bitrate / 1000.0,  # kbit/s
                            "fps": meta.fps,
                        }
                    ],
                },
                "I23": {"streamId": 42, "stalling": []},
                "IGen": {
                    "device": device_map.get(self.display_size, "mobile"),
                    "displaySize": f"{meta.width}x{meta.height}",
                    "viewingDistance": 0,
                },
            }

            result = self._p1203_cls(input_json).calculate_complete()
            mos = result.get("O46")
            if mos is not None:
                return float(max(1.0, min(5.0, mos)))
            return None
        except Exception as e:
            logger.warning(f"P.1203 computation failed: {e}")
            return None
