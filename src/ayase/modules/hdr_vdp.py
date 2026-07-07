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
    metric_groups = {
        "hdr_vdp": "hdr",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 5)
        self._ml_available = False
        self._backend = None

    def setup(self) -> None:
        # HDR-VDP requires the real hdrvdp Python bindings/CLI.
        try:
            import hdrvdp  # noqa: F401
            self._backend = "python"
            self._ml_available = True
            logger.info("HDR-VDP module initialised (Python bindings)")
            return
        except ImportError:
            pass

        self._backend = "unavailable"
        self._ml_available = False
        logger.warning("HDR-VDP unavailable: hdrvdp Python bindings are not installed.")

    def compute_reference_score(
        self, sample_path: Path, reference_path: Path
    ) -> Optional[float]:
        if self._backend != "python":
            return None
        ref_img = cv2.imread(str(reference_path))
        dist_img = cv2.imread(str(sample_path))
        if ref_img is None or dist_img is None:
            return None

        h = min(ref_img.shape[0], dist_img.shape[0])
        w = min(ref_img.shape[1], dist_img.shape[1])
        ref_img = cv2.resize(ref_img, (w, h))
        dist_img = cv2.resize(dist_img, (w, h))

        return self._compute_hdrvdp(ref_img, dist_img)

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
                s = self._compute_hdrvdp(ref_r, dist_r)
                if s is not None:
                    scores.append(s)
            idx += 1
        ref_cap.release()
        dist_cap.release()
        return float(np.mean(scores)) if scores else None
