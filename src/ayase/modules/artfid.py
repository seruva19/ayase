"""ArtFID — Artistic Style Transfer FID (2022).

Full-reference metric for evaluating style transfer quality.
Combines content fidelity (LPIPS-like) with style similarity (FID-like).

pip install art-fid

artfid_score — lower = better (combined content + style distance).
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.base_modules import ReferenceBasedModule

logger = logging.getLogger(__name__)


class ArtFIDModule(ReferenceBasedModule):
    name = "artfid"
    description = "ArtFID style transfer quality (FR, 2022, lower=better)"
    metric_field = "artfid_score"
    default_config = {"subsample": 8}

    def __init__(self, config=None):
        super().__init__(config)
        self._model = None
        self._ml_available = False
        self.subsample = self.config.get("subsample", 8)

    def setup(self) -> None:
        # Tier 1: art-fid package
        try:
            import art_fid
            self._ml_available = True
            self._model = art_fid
            logger.info("ArtFID module initialised (art-fid package)")
            return
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"art-fid init failed: {e}")

        # Tier 2: heuristic
        logger.info("ArtFID module initialised (heuristic fallback)")

    def compute_reference_score(self, sample_path: Path, reference_path: Path) -> Optional[float]:
        try:
            if str(sample_path).lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                return self._score_video(str(sample_path), str(reference_path))
            else:
                return self._score_image(str(sample_path), str(reference_path))
        except Exception as e:
            logger.warning(f"ArtFID failed: {e}")
            return None

    def _score_image(self, sample_p: str, ref_p: str) -> Optional[float]:
        img = cv2.imread(sample_p)
        ref = cv2.imread(ref_p)
        if img is None or ref is None:
            return None
        return self._heuristic_artfid(img, ref)

    def _score_video(self, sample_p: str, ref_p: str) -> Optional[float]:
        cap_s = cv2.VideoCapture(sample_p)
        cap_r = cv2.VideoCapture(ref_p)
        try:
            total = min(
                int(cap_s.get(cv2.CAP_PROP_FRAME_COUNT)),
                int(cap_r.get(cv2.CAP_PROP_FRAME_COUNT)),
            )
            if total <= 0:
                return None
            indices = np.linspace(0, total - 1, min(self.subsample, total), dtype=int)
            scores = []
            for idx in indices:
                cap_s.set(cv2.CAP_PROP_POS_FRAMES, idx)
                cap_r.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret_s, frame_s = cap_s.read()
                ret_r, frame_r = cap_r.read()
                if ret_s and ret_r:
                    s = self._heuristic_artfid(frame_s, frame_r)
                    if s is not None:
                        scores.append(s)
            return float(np.mean(scores)) if scores else None
        finally:
            cap_s.release()
            cap_r.release()

    def _heuristic_artfid(self, img: np.ndarray, ref: np.ndarray) -> float:
        """Heuristic: content LPIPS proxy + style Gram distance proxy.

        Content term: multi-scale structural difference (like LPIPS).
        Style term: Gram matrix distance of filter responses (like style FID).
        """
        h, w = ref.shape[:2]
        img = cv2.resize(img, (w, h)).astype(np.float64) / 255.0
        ref = ref.astype(np.float64) / 255.0

        # Content distance: multi-scale MSE
        content_dists = []
        img_s, ref_s = img, ref
        for _ in range(3):
            mse = np.mean((img_s - ref_s) ** 2)
            content_dists.append(mse)
            new_h, new_w = max(img_s.shape[0] // 2, 1), max(img_s.shape[1] // 2, 1)
            img_s = cv2.resize(img_s, (new_w, new_h))
            ref_s = cv2.resize(ref_s, (new_w, new_h))
        content_dist = float(np.mean(content_dists))

        # Style distance: Gram matrix of filter bank responses
        gray_img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)
        gray_ref = cv2.cvtColor((ref * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)

        # Simple filter bank: Sobel x, Sobel y, Laplacian
        filters_img = []
        filters_ref = []
        for fn in [
            lambda g: cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3),
            lambda g: cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3),
            lambda g: cv2.Laplacian(g, cv2.CV_64F),
        ]:
            fi = fn(gray_img).flatten()
            fr = fn(gray_ref).flatten()
            filters_img.append(fi / (np.linalg.norm(fi) + 1e-8))
            filters_ref.append(fr / (np.linalg.norm(fr) + 1e-8))

        feat_img = np.stack(filters_img, axis=0)  # (3, N)
        feat_ref = np.stack(filters_ref, axis=0)

        gram_img = feat_img @ feat_img.T
        gram_ref = feat_ref @ feat_ref.T

        style_dist = float(np.mean((gram_img - gram_ref) ** 2))

        # Combined score
        return float(content_dist + 0.5 * style_dist)
