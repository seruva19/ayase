"""HDR Metadata (MaxFALL / MaxCLL) module.

Computes HDR static metadata values:
  MaxFALL — Maximum Frame-Average Light Level (nits)
  MaxCLL  — Maximum Content Light Level (nits)

These are critical HDR10 metadata values used for tone-mapping.
Works on any video by analyzing pixel luminance values.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


def _bt709_luminance(bgr: np.ndarray) -> np.ndarray:
    """Convert BGR to relative luminance using BT.709 coefficients."""
    b, g, r = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


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


class HDRMetadataModule(PipelineModule):
    name = "hdr_metadata"
    description = "MaxFALL + MaxCLL HDR static metadata analysis"
    default_config = {
        "subsample": 3,
        "peak_nits": 10000.0,  # Assumed peak luminance for 10-bit PQ
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 3)
        self.peak_nits = self.config.get("peak_nits", 10000.0)
        self._ml_available = True  # Pure OpenCV

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video:
            return sample

        try:
            max_fall, max_cll = self._analyze_video(sample.path)
            if max_fall is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.max_fall = max_fall
            sample.quality_metrics.max_cll = max_cll
            logger.debug(f"HDR metadata for {sample.path.name}: MaxFALL={max_fall:.1f} MaxCLL={max_cll:.1f}")
        except Exception as e:
            logger.error(f"HDR metadata failed: {e}")
        return sample

    def _analyze_video(self, path: Path):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None, None

        frame_averages = []
        pixel_max = 0.0
        idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % self.subsample == 0:
                frame_f = frame.astype(np.float32)
                lum = _bt709_luminance(frame_f)

                # Detect HDR: uint16/float32 source or high dynamic range values
                is_hdr = frame.dtype in (np.uint16, np.float32, np.float64) or frame.max() > 255

                if is_hdr:
                    # Apply PQ EOTF for proper HDR10 luminance mapping
                    signal = lum / (65535.0 if frame.dtype == np.uint16 else max(float(lum.max()), 1.0))
                    lum_nits = _pq_eotf(signal)
                else:
                    # SDR content: use raw luminance values directly (cd/m^2 proxy)
                    lum_nits = lum

                frame_avg = float(np.mean(lum_nits))
                frame_max = float(np.max(lum_nits))
                frame_averages.append(frame_avg)
                pixel_max = max(pixel_max, frame_max)
            idx += 1

        cap.release()

        if not frame_averages:
            return None, None

        max_fall = float(max(frame_averages))
        max_cll = pixel_max
        return max_fall, max_cll
