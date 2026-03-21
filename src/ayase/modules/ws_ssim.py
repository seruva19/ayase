"""WS-SSIM — Weighted Spherical SSIM. ws_ssim — 0-1, higher = better"""
import logging, cv2, numpy as np
from pathlib import Path
from typing import Optional
from ayase.models import Sample, QualityMetrics
from ayase.base_modules import ReferenceBasedModule
logger = logging.getLogger(__name__)

class WSSSIMModule(ReferenceBasedModule):
    name = "ws_ssim"; description = "WS-SSIM weighted spherical SSIM"; metric_field = "ws_ssim"; default_config = {"subsample": 8}

    def _compute_frame_pair(self, img_gray: np.ndarray, ref_gray: np.ndarray) -> Optional[float]:
        """Compute WS-SSIM for a single grayscale frame pair."""
        try:
            h, w = img_gray.shape
            ref_gray = cv2.resize(ref_gray, (w, h))
            img_f, ref_f = img_gray.astype(np.float64), ref_gray.astype(np.float64)
            lat = np.linspace(-np.pi/2, np.pi/2, h).reshape(-1, 1)
            weights = np.cos(lat); weights = np.broadcast_to(weights, (h, w))
            C1, C2 = 6.5025, 58.5225
            mu1 = cv2.GaussianBlur(img_f, (11,11), 1.5)
            mu2 = cv2.GaussianBlur(ref_f, (11,11), 1.5)
            s1 = np.maximum(cv2.GaussianBlur(img_f**2, (11,11), 1.5) - mu1**2, 0)
            s2 = np.maximum(cv2.GaussianBlur(ref_f**2, (11,11), 1.5) - mu2**2, 0)
            s12 = cv2.GaussianBlur(img_f*ref_f, (11,11), 1.5) - mu1*mu2
            ssim_map = ((2*mu1*mu2+C1)*(2*s12+C2))/((mu1**2+mu2**2+C1)*(s1+s2+C2))
            return float(np.sum(weights*ssim_map)/np.sum(weights))
        except Exception as e:
            logger.warning(f"WS-SSIM frame failed: {e}"); return None

    def compute_reference_score(self, sample_path: Path, reference_path: Path) -> Optional[float]:
        """Compute WS-SSIM for images or videos (averaging per-frame scores)."""
        ext = sample_path.suffix.lower()
        if ext in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
            return self._compute_video(sample_path, reference_path)
        # Image path
        try:
            img = cv2.imread(str(sample_path), cv2.IMREAD_GRAYSCALE)
            ref = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
            if img is None or ref is None: return None
            return self._compute_frame_pair(img, ref)
        except Exception as e:
            logger.warning(f"WS-SSIM failed: {e}"); return None

    def _compute_video(self, sample_path: Path, reference_path: Path) -> Optional[float]:
        """Iterate video frames, compute per-frame WS-SSIM, return average."""
        cap_s = cv2.VideoCapture(str(sample_path))
        cap_r = cv2.VideoCapture(str(reference_path))
        if not cap_s.isOpened() or not cap_r.isOpened():
            cap_s.release(); cap_r.release()
            return None

        subsample = self.config.get("subsample", 8)
        total_s = int(cap_s.get(cv2.CAP_PROP_FRAME_COUNT))
        total_r = int(cap_r.get(cv2.CAP_PROP_FRAME_COUNT))
        total = min(total_s, total_r)
        if total <= 0:
            cap_s.release(); cap_r.release()
            return None
        indices = list(range(0, total, max(1, total // subsample)))[:subsample]

        scores = []
        for idx in indices:
            cap_s.set(cv2.CAP_PROP_POS_FRAMES, idx)
            cap_r.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret_s, frame_s = cap_s.read()
            ret_r, frame_r = cap_r.read()
            if not ret_s or not ret_r:
                continue
            gray_s = cv2.cvtColor(frame_s, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
            s = self._compute_frame_pair(gray_s, gray_r)
            if s is not None:
                scores.append(s)

        cap_s.release(); cap_r.release()
        return float(np.mean(scores)) if scores else None
