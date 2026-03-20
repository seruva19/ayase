"""WS-SSIM — Weighted Spherical SSIM. ws_ssim — 0-1, higher = better"""
import logging, cv2, numpy as np
from pathlib import Path
from typing import Optional
from ayase.models import Sample, QualityMetrics
from ayase.base_modules import ReferenceBasedModule
logger = logging.getLogger(__name__)

class WSSSIMModule(ReferenceBasedModule):
    name = "ws_ssim"; description = "WS-SSIM weighted spherical SSIM"; metric_field = "ws_ssim"; default_config = {}
    def compute_reference_score(self, sample_path: Path, reference_path: Path) -> Optional[float]:
        try:
            img = cv2.imread(str(sample_path), cv2.IMREAD_GRAYSCALE)
            ref = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
            if img is None or ref is None: return None
            h, w = img.shape; ref = cv2.resize(ref, (w, h))
            img_f, ref_f = img.astype(np.float64), ref.astype(np.float64)
            lat = np.linspace(-np.pi/2, np.pi/2, h).reshape(-1, 1)
            weights = np.cos(lat); weights = np.broadcast_to(weights, (h, w))
            # Weighted SSIM
            C1, C2 = 6.5025, 58.5225
            mu1 = cv2.GaussianBlur(img_f, (11,11), 1.5)
            mu2 = cv2.GaussianBlur(ref_f, (11,11), 1.5)
            s1 = cv2.GaussianBlur(img_f**2, (11,11), 1.5) - mu1**2
            s2 = cv2.GaussianBlur(ref_f**2, (11,11), 1.5) - mu2**2
            s12 = cv2.GaussianBlur(img_f*ref_f, (11,11), 1.5) - mu1*mu2
            ssim_map = ((2*mu1*mu2+C1)*(2*s12+C2))/((mu1**2+mu2**2+C1)*(s1+s2+C2))
            return float(np.sum(weights*ssim_map)/np.sum(weights))
        except Exception as e:
            logger.warning(f"WS-SSIM failed: {e}"); return None
