"""Watermark Robustness module.

Estimates invisible watermark presence and strength:

  watermark_strength — 0-1 (higher = stronger watermark detected)

Detection methods:
  1. DCT-domain analysis: invisible watermarks often embed energy
     in specific frequency sub-bands of the DCT.
  2. Bit-plane analysis: watermarks may be visible in LSB planes.
  3. If ``invisible-watermark`` library is available, uses its
     decoder for known watermark formats (DwtDct, RivaGAN).
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class WatermarkRobustnessModule(PipelineModule):
    name = "watermark_robustness"
    description = "Invisible watermark detection and strength estimation"
    default_config = {
        "subsample": 15,
        "max_frames": 30,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 15)
        self.max_frames = self.config.get("max_frames", 30)
        self._decoder = None
        self._decoder_available = False

    def setup(self) -> None:
        try:
            from imwatermark import WatermarkDecoder
            self._decoder = WatermarkDecoder("bytes", 32)
            self._decoder_available = True
            logger.info("Watermark robustness: imwatermark decoder available")
        except ImportError:
            logger.info("imwatermark not installed, using heuristic detection only")
        except Exception as e:
            logger.warning(f"Watermark decoder init failed: {e}")

    # ------------------------------------------------------------------
    # Heuristic detection
    # ------------------------------------------------------------------

    @staticmethod
    def _dct_watermark_score(gray: np.ndarray) -> float:
        """Detect watermark energy in DCT mid-frequency bands.

        Invisible watermarks typically embed information in mid-
        frequency DCT coefficients.  We check if mid-band energy
        is anomalously high relative to high-frequency noise.

        Returns 0-1 (higher = more likely watermarked).
        """
        h, w = gray.shape
        # Process in 8x8 blocks
        block_size = 8
        n_blocks_h = h // block_size
        n_blocks_w = w // block_size

        if n_blocks_h < 2 or n_blocks_w < 2:
            return 0.0

        mid_energies = []
        high_energies = []

        gray_f = gray.astype(np.float32)

        for by in range(0, n_blocks_h * block_size, block_size):
            for bx in range(0, n_blocks_w * block_size, block_size):
                block = gray_f[by:by + block_size, bx:bx + block_size]
                dct = cv2.dct(block)

                # Mid-frequency: positions (2,2) to (5,5) roughly
                mid = np.abs(dct[2:6, 2:6]).mean()
                # High-frequency: positions (6,6) to (7,7)
                high = np.abs(dct[6:, 6:]).mean()

                mid_energies.append(mid)
                high_energies.append(high)

        avg_mid = float(np.mean(mid_energies))
        avg_high = float(np.mean(high_energies)) + 1e-6

        # Ratio: watermarked images have higher mid-to-high ratio
        ratio = avg_mid / avg_high

        # Natural images: ratio ~2-5.  Watermarked: ratio often > 6
        return float(np.clip((ratio - 3.0) / 5.0, 0, 1))

    @staticmethod
    def _lsb_uniformity_score(gray: np.ndarray) -> float:
        """Check if LSB plane has unusual uniformity (watermark carrier).

        Natural images have random-looking LSBs.  Watermarked images
        may have structured LSB patterns.

        Returns 0-1 (higher = more structured = likely watermarked).
        """
        lsb = gray & 1  # Extract LSB

        # Compute local uniformity via block-wise entropy
        h, w = lsb.shape
        block = 16
        entropies = []

        for by in range(0, h - block, block):
            for bx in range(0, w - block, block):
                patch = lsb[by:by + block, bx:bx + block].flatten()
                p = patch.mean()
                if 0 < p < 1:
                    ent = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
                else:
                    ent = 0.0
                entropies.append(ent)

        if not entropies:
            return 0.0

        avg_entropy = float(np.mean(entropies))
        # Natural: entropy ~1.0 (random).  Watermark: slightly below.
        # Very low entropy (<0.8) suggests structured LSB
        return float(np.clip((1.0 - avg_entropy) * 3.0, 0, 1))

    # ------------------------------------------------------------------
    # Library-based detection
    # ------------------------------------------------------------------

    def _decode_watermark(self, frame_bgr: np.ndarray) -> float:
        """Try to decode an invisible watermark.  Returns 1 if found."""
        if not self._decoder_available:
            return 0.0
        try:
            wm = self._decoder.decode(frame_bgr, "dwtDct")
            # If decoded bytes are non-zero, watermark likely present
            if wm is not None and any(b != 0 for b in wm):
                return 1.0
        except Exception:
            pass
        return 0.0

    # ------------------------------------------------------------------
    # Combined
    # ------------------------------------------------------------------

    def _score_frame(self, frame_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        dct_score = self._dct_watermark_score(gray)
        lsb_score = self._lsb_uniformity_score(gray)
        lib_score = self._decode_watermark(frame_bgr)

        if lib_score > 0.5:
            # Library confirmed watermark
            return max(0.8, (dct_score + lsb_score + lib_score) / 3)

        return 0.5 * dct_score + 0.5 * lsb_score

    # ------------------------------------------------------------------
    # Process
    # ------------------------------------------------------------------

    def process(self, sample: Sample) -> Sample:
        try:
            if sample.is_video:
                score = self._process_video(sample.path)
            else:
                score = self._process_image(sample.path)

            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.watermark_strength = score

            logger.debug(f"Watermark strength for {sample.path.name}: {score:.3f}")

        except Exception as e:
            logger.error(f"Watermark detection failed for {sample.path}: {e}")

        return sample

    def _process_image(self, path: Path) -> Optional[float]:
        img = cv2.imread(str(path))
        if img is None:
            return None
        return self._score_frame(img)

    def _process_video(self, path: Path) -> Optional[float]:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None

        scores = []
        idx = 0

        while idx < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % self.subsample == 0:
                scores.append(self._score_frame(frame))
            idx += 1

        cap.release()
        return float(np.mean(scores)) if scores else None
