"""Spherical PSNR metrics — S-PSNR, WS-PSNR, CPP-PSNR (MPEG/JVET).

GitHub: https://github.com/Samsung/360tools

s_psnr, ws_psnr, cpp_psnr — all dB, higher = better
"""
import logging, cv2, numpy as np
from pathlib import Path
from typing import Optional
from ayase.models import Sample, QualityMetrics
from ayase.base_modules import ReferenceBasedModule
logger = logging.getLogger(__name__)

class SphericalPSNRModule(ReferenceBasedModule):
    name = "spherical_psnr"
    description = "S-PSNR/WS-PSNR/CPP-PSNR spherical PSNR (MPEG/JVET)"
    metric_field = None  # We set multiple fields manually
    default_config = {"subsample": 8}

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)

    def process(self, sample: Sample) -> Sample:
        ref = getattr(sample, "reference_path", None)
        if ref is None: return sample
        if not isinstance(ref, Path): ref = Path(ref)
        if not ref.exists(): return sample
        try:
            scores = self._compute(str(sample.path), str(ref))
            if scores:
                if sample.quality_metrics is None: sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.s_psnr = scores["s_psnr"]
                sample.quality_metrics.ws_psnr = scores["ws_psnr"]
                sample.quality_metrics.cpp_psnr = scores["cpp_psnr"]
        except Exception as e:
            logger.warning(f"SphericalPSNR failed: {e}")
        return sample

    def compute_reference_score(self, sample_path: Path, reference_path: Path) -> Optional[float]:
        scores = self._compute(str(sample_path), str(reference_path))
        return scores.get("ws_psnr") if scores else None

    def _compute(self, sample_p, ref_p):
        """Heuristic: equirectangular area-weighted PSNR."""
        img = cv2.imread(sample_p, cv2.IMREAD_GRAYSCALE)
        ref = cv2.imread(ref_p, cv2.IMREAD_GRAYSCALE)
        if img is None or ref is None: return None
        h, w = img.shape
        ref = cv2.resize(ref, (w, h))
        img_f = img.astype(np.float64)
        ref_f = ref.astype(np.float64)
        diff_sq = (img_f - ref_f) ** 2

        # Standard PSNR (proxy for S-PSNR)
        mse = np.mean(diff_sq)
        s_psnr = 10 * np.log10(255**2 / max(mse, 1e-10))

        # WS-PSNR: weight by cos(latitude)
        lat = np.linspace(-np.pi/2, np.pi/2, h).reshape(-1, 1)
        weights = np.cos(lat)
        weights = np.broadcast_to(weights, (h, w))
        ws_mse = np.sum(weights * diff_sq) / np.sum(weights)
        ws_psnr = 10 * np.log10(255**2 / max(ws_mse, 1e-10))

        # CPP-PSNR: Craster Parabolic Projection weighting
        # Weight for latitude theta is cos(theta) * (1 - sin^2(theta)/3)
        cpp_weights = np.cos(lat) * (1.0 - np.sin(lat) ** 2 / 3.0)
        cpp_weights = np.broadcast_to(cpp_weights, (h, w))
        cpp_mse = np.sum(cpp_weights * diff_sq) / np.sum(cpp_weights)
        cpp_psnr = 10 * np.log10(255**2 / max(cpp_mse, 1e-10))

        return {"s_psnr": float(s_psnr), "ws_psnr": float(ws_psnr), "cpp_psnr": float(cpp_psnr)}
