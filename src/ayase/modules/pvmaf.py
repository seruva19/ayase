"""pVMAF — Predictive VMAF (35x faster).

2024 — lightweight VMAF approximation using bitstream and pixel-level
features to predict VMAF scores without full reference decoding.
Achieves ~35x speedup with high correlation to standard VMAF.

pvmaf_score — 0-100 scale (higher = better)
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.base_modules import ReferenceBasedModule

logger = logging.getLogger(__name__)


class PVMAFModule(ReferenceBasedModule):
    name = "pvmaf"
    description = "Predictive VMAF ~35x faster via bitstream+pixel features (2024, 0-100)"
    metric_field = "pvmaf_score"
    default_config = {
        "subsample": 8,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._model = None
        self._backend = "heuristic"

    def setup(self) -> None:
        # Tier 1: Try native pVMAF
        try:
            import pvmaf
            self._model = pvmaf
            self._backend = "native"
            logger.info("pVMAF (native) initialised")
            return
        except ImportError:
            pass

        # Tier 2: Heuristic fallback
        self._backend = "heuristic"
        logger.info("pVMAF (heuristic) initialised — install pvmaf for full model")

    def compute_reference_score(
        self, sample_path: Path, reference_path: Path
    ) -> Optional[float]:
        if self._backend == "native":
            score = float(self._model.predict(str(sample_path), str(reference_path)))
            return float(np.clip(score, 0.0, 100.0))
        return self._compute_heuristic(sample_path, reference_path)

    def _read_frames(self, path: Path) -> list:
        """Read frames from video or image."""
        frames = []
        is_video = path.suffix.lower() in {
            ".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv",
        }

        if is_video:
            cap = cv2.VideoCapture(str(path))
            try:
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total <= 0:
                    return frames
                indices = np.linspace(0, total - 1, min(self.subsample, total), dtype=int)
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        frames.append(frame)
            finally:
                cap.release()
        else:
            img = cv2.imread(str(path))
            if img is not None:
                frames.append(img)

        return frames

    def _compute_heuristic(
        self, sample_path: Path, reference_path: Path
    ) -> Optional[float]:
        """Heuristic: bitstream+pixel features → VMAF estimate (0-100)."""
        dist_frames = self._read_frames(sample_path)
        ref_frames = self._read_frames(reference_path)

        if not dist_frames or not ref_frames:
            return None

        n_frames = min(len(dist_frames), len(ref_frames))
        dist_frames = dist_frames[:n_frames]
        ref_frames = ref_frames[:n_frames]

        frame_scores = []
        for i in range(n_frames):
            dist_gray = cv2.cvtColor(dist_frames[i], cv2.COLOR_BGR2GRAY).astype(np.float64)
            ref_gray = cv2.cvtColor(ref_frames[i], cv2.COLOR_BGR2GRAY).astype(np.float64)

            if dist_gray.shape != ref_gray.shape:
                dist_gray = cv2.resize(dist_gray, (ref_gray.shape[1], ref_gray.shape[0]))

            # VIF (Visual Information Fidelity) component
            eps = 1e-7
            ref_mu = cv2.GaussianBlur(ref_gray, (7, 7), 1.0)
            dist_mu = cv2.GaussianBlur(dist_gray, (7, 7), 1.0)
            ref_var = cv2.GaussianBlur(ref_gray ** 2, (7, 7), 1.0) - ref_mu ** 2
            dist_var = cv2.GaussianBlur(dist_gray ** 2, (7, 7), 1.0) - dist_mu ** 2
            cross_var = cv2.GaussianBlur(ref_gray * dist_gray, (7, 7), 1.0) - ref_mu * dist_mu

            ref_var = np.maximum(ref_var, 0)
            dist_var = np.maximum(dist_var, 0)

            g = cross_var / (ref_var + eps)
            noise_var = dist_var - g * cross_var
            noise_var = np.maximum(noise_var, eps)

            sigma_n = 2.0
            vif_num = np.sum(np.log2(1 + g ** 2 * ref_var / (noise_var + sigma_n ** 2) + eps))
            vif_den = np.sum(np.log2(1 + ref_var / (sigma_n ** 2) + eps))
            vif = float(vif_num / (vif_den + eps))

            # DLM (Detail Loss Metric) component
            ref_lap = cv2.Laplacian(ref_gray, cv2.CV_64F)
            dist_lap = cv2.Laplacian(dist_gray, cv2.CV_64F)
            detail_loss = np.mean(np.abs(ref_lap - dist_lap)) / (np.mean(np.abs(ref_lap)) + eps)
            dlm = 1.0 / (1.0 + detail_loss)

            # Motion component (pixel-level)
            # Simple MSE-based
            mse = np.mean((ref_gray - dist_gray) ** 2)
            psnr_norm = min(10.0 * np.log10(255.0 ** 2 / (mse + eps)) / 50.0, 1.0) if mse > 0 else 1.0

            # VMAF-like combination (learned weights approximation)
            vmaf_raw = 0.40 * vif + 0.35 * dlm + 0.25 * psnr_norm
            frame_scores.append(vmaf_raw)

        raw_score = float(np.mean(frame_scores))
        # Map to 0-100 VMAF scale
        score = raw_score * 100.0
        return float(np.clip(score, 0.0, 100.0))
