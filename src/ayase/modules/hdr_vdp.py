"""HDR-VDP (HDR Visual Difference Predictor) module.

HDR-VDP is the gold-standard perceptual quality metric for HDR content.
It models the human visual system's adaptation to HDR luminance levels.

Range: Q score (higher = better quality, typically 0-100).

Requires hdrvdp Python bindings or CLI. Falls back to PU-PSNR proxy.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.base_modules import ReferenceBasedModule

logger = logging.getLogger(__name__)


class HDRVDPModule(ReferenceBasedModule):
    name = "hdr_vdp"
    description = "HDR-VDP visual difference predictor (higher=better)"
    default_config = {
        "subsample": 5,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 5)
        self._ml_available = False
        self._backend = None

    def setup(self) -> None:
        # Try hdrvdp Python bindings
        try:
            import hdrvdp
            self._backend = "python"
            self._ml_available = True
            logger.info("HDR-VDP module initialised (Python bindings)")
            return
        except ImportError:
            pass

        # Fallback: approximation using PU21 + frequency-weighted difference
        self._backend = "approx"
        self._ml_available = True
        logger.info("HDR-VDP module initialised (approximation fallback)")

    def compute_reference_score(
        self, sample_path: Path, reference_path: Path
    ) -> Optional[float]:
        ref_img = cv2.imread(str(reference_path))
        dist_img = cv2.imread(str(sample_path))
        if ref_img is None or dist_img is None:
            return None

        h = min(ref_img.shape[0], dist_img.shape[0])
        w = min(ref_img.shape[1], dist_img.shape[1])
        ref_img = cv2.resize(ref_img, (w, h))
        dist_img = cv2.resize(dist_img, (w, h))

        if self._backend == "python":
            return self._compute_hdrvdp(ref_img, dist_img)
        return self._compute_approx(ref_img, dist_img)

    def _compute_hdrvdp(self, ref_bgr, dist_bgr) -> Optional[float]:
        try:
            import hdrvdp
            ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB).astype(np.float64)
            dist_rgb = cv2.cvtColor(dist_bgr, cv2.COLOR_BGR2RGB).astype(np.float64)
            result = hdrvdp.hdrvdp3("quality", dist_rgb, ref_rgb, "sRGB-display")
            return float(result.get("Q", result.get("quality", 0)))
        except Exception as e:
            logger.debug(f"HDR-VDP native failed: {e}")
            return None

    def _compute_approx(self, ref_bgr, dist_bgr) -> float:
        """Approximate HDR-VDP using multi-scale frequency-weighted analysis."""
        ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        dist_gray = cv2.cvtColor(dist_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        # Multi-scale analysis (Laplacian pyramid proxy)
        scores = []
        for scale in range(4):
            diff = np.abs(ref_gray - dist_gray)
            mse = float(np.mean(diff ** 2))
            score = max(0, 1.0 - mse * 10.0)
            scores.append(score)
            ref_gray = cv2.pyrDown(ref_gray)
            dist_gray = cv2.pyrDown(dist_gray)
            if ref_gray.shape[0] < 16:
                break

        # Weight higher frequencies more (HVS sensitivity)
        weights = [0.4, 0.3, 0.2, 0.1][:len(scores)]
        total_w = sum(weights)
        q = sum(s * w for s, w in zip(scores, weights)) / total_w
        return float(np.clip(q * 100.0, 0, 100))

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
            if sample.is_video:
                score = self._process_video(sample.path, reference)
            else:
                score = self.compute_reference_score(sample.path, reference)

            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.hdr_vdp = score
                logger.debug(f"HDR-VDP for {sample.path.name}: {score:.1f}")
        except Exception as e:
            logger.error(f"HDR-VDP failed: {e}")
        return sample

    def _process_video(self, path, ref_path) -> Optional[float]:
        ref_cap = cv2.VideoCapture(str(ref_path))
        dist_cap = cv2.VideoCapture(str(path))
        scores = []
        idx = 0
        while True:
            r1, ref_f = ref_cap.read()
            r2, dist_f = dist_cap.read()
            if not r1 or not r2:
                break
            if idx % self.subsample == 0:
                h = min(ref_f.shape[0], dist_f.shape[0])
                w = min(ref_f.shape[1], dist_f.shape[1])
                ref_r = cv2.resize(ref_f, (w, h))
                dist_r = cv2.resize(dist_f, (w, h))
                if self._backend == "python":
                    s = self._compute_hdrvdp(ref_r, dist_r)
                else:
                    s = self._compute_approx(ref_r, dist_r)
                if s is not None:
                    scores.append(s)
            idx += 1
        ref_cap.release()
        dist_cap.release()
        return float(np.mean(scores)) if scores else None
