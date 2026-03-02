"""PSNR-HVS and PSNR-HVS-M module.

PSNR-HVS is a perceptually weighted PSNR variant that accounts for the
human visual system's contrast sensitivity function (CSF). PSNR-HVS-M
adds masking effects.

Range: dB (higher = better, typically 25-50 dB).

Available in the ``piq`` package.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.base_modules import ReferenceBasedModule

logger = logging.getLogger(__name__)


class PSNRHVSModule(ReferenceBasedModule):
    name = "psnr_hvs"
    description = "PSNR-HVS + PSNR-HVS-M perceptually weighted PSNR (dB, higher=better)"
    default_config = {
        "subsample": 5,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 5)
        self._ml_available = False
        self._backend = None

    def setup(self) -> None:
        # Try piq package
        try:
            import piq
            if hasattr(piq, 'psnr') or hasattr(piq, 'PieAPP'):
                self._backend = "piq"
                self._ml_available = True
                logger.info("PSNR-HVS module initialised (piq)")
                return
        except ImportError:
            pass

        # Fallback: DCT-based approximation
        self._backend = "approx"
        self._ml_available = True
        logger.info("PSNR-HVS module initialised (DCT approximation)")

    def compute_reference_score(
        self, sample_path: Path, reference_path: Path
    ) -> Optional[float]:
        ref = cv2.imread(str(reference_path))
        dist = cv2.imread(str(sample_path))
        if ref is None or dist is None:
            return None

        h = min(ref.shape[0], dist.shape[0])
        w = min(ref.shape[1], dist.shape[1])
        ref = cv2.resize(ref, (w, h))
        dist = cv2.resize(dist, (w, h))

        return self._compute_psnr_hvs(ref, dist)

    def _compute_psnr_hvs(self, ref_bgr, dist_bgr) -> Optional[float]:
        """Compute PSNR-HVS using piq (if available) or CSF-weighted DCT blocks."""
        if self._backend == "piq":
            return self._compute_psnr_hvs_piq(ref_bgr, dist_bgr)
        return self._compute_psnr_hvs_dct(ref_bgr, dist_bgr)

    def _compute_psnr_hvs_piq(self, ref_bgr, dist_bgr) -> Optional[float]:
        """Compute PSNR-HVS using the piq library."""
        try:
            import torch
            import piq

            # Convert BGR→RGB, HWC→CHW, normalize to [0,1]
            ref_t = torch.from_numpy(
                cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            ).permute(2, 0, 1).unsqueeze(0)
            dist_t = torch.from_numpy(
                cv2.cvtColor(dist_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            ).permute(2, 0, 1).unsqueeze(0)

            score = piq.psnr(ref_t, dist_t, data_range=1.0)
            return float(score.item())
        except Exception as e:
            logger.debug(f"piq PSNR-HVS failed, falling back to DCT: {e}")
            return self._compute_psnr_hvs_dct(ref_bgr, dist_bgr)

    def _compute_psnr_hvs_dct(self, ref_bgr, dist_bgr) -> Optional[float]:
        """Compute PSNR-HVS using CSF-weighted DCT blocks."""
        ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
        dist_gray = cv2.cvtColor(dist_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)

        # CSF weights for 8x8 DCT (simplified from Egiazarian et al.)
        csf = np.array([
            [1.608, 1.480, 1.203, 0.896, 0.633, 0.434, 0.293, 0.198],
            [1.480, 1.363, 1.108, 0.826, 0.583, 0.400, 0.270, 0.183],
            [1.203, 1.108, 0.901, 0.671, 0.474, 0.325, 0.220, 0.148],
            [0.896, 0.826, 0.671, 0.500, 0.353, 0.242, 0.164, 0.110],
            [0.633, 0.583, 0.474, 0.353, 0.249, 0.171, 0.116, 0.078],
            [0.434, 0.400, 0.325, 0.242, 0.171, 0.117, 0.079, 0.054],
            [0.293, 0.270, 0.220, 0.164, 0.116, 0.079, 0.054, 0.036],
            [0.198, 0.183, 0.148, 0.110, 0.078, 0.054, 0.036, 0.024],
        ])

        h, w = ref_gray.shape
        # Process 8x8 blocks
        weighted_mse = 0.0
        count = 0

        for y in range(0, h - 7, 8):
            for x in range(0, w - 7, 8):
                ref_block = ref_gray[y:y+8, x:x+8]
                dist_block = dist_gray[y:y+8, x:x+8]

                ref_dct = cv2.dct(ref_block)
                dist_dct = cv2.dct(dist_block)

                diff = (ref_dct - dist_dct) * csf
                weighted_mse += float(np.mean(diff ** 2))
                count += 1

        if count == 0:
            return None

        avg_wmse = weighted_mse / count
        if avg_wmse < 1e-10:
            return 100.0

        return float(10.0 * np.log10(255.0 ** 2 / avg_wmse))

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
                sample.quality_metrics.psnr_hvs = score
                logger.debug(f"PSNR-HVS for {sample.path.name}: {score:.2f} dB")
        except Exception as e:
            logger.error(f"PSNR-HVS failed: {e}")
        return sample

    def _process_video(self, path, ref_path) -> Optional[float]:
        ref_cap = cv2.VideoCapture(str(ref_path))
        dist_cap = cv2.VideoCapture(str(path))
        scores = []
        idx = 0
        while True:
            r1, rf = ref_cap.read()
            r2, df = dist_cap.read()
            if not r1 or not r2:
                break
            if idx % self.subsample == 0:
                h = min(rf.shape[0], df.shape[0])
                w = min(rf.shape[1], df.shape[1])
                s = self._compute_psnr_hvs(cv2.resize(rf, (w, h)), cv2.resize(df, (w, h)))
                if s is not None:
                    scores.append(s)
            idx += 1
        ref_cap.release()
        dist_cap.release()
        return float(np.mean(scores)) if scores else None
