"""V-BLIINDS — Video BLind Image Integrity Notator using DCT Statistics.

Saad et al. 2013 — blind NR-VQA using DCT-domain natural scene
statistics (NSS). Extracts statistical features from DCT coefficients
of local patches and uses an SVR-like model for quality prediction.

vbliinds_score — higher = better quality
"""

import logging
import cv2
import numpy as np
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


def _dct_block_features(gray: np.ndarray, block_size: int = 8) -> np.ndarray:
    """Extract DCT coefficient statistics from non-overlapping blocks."""
    h, w = gray.shape
    h_blocks = h // block_size
    w_blocks = w // block_size
    if h_blocks == 0 or w_blocks == 0:
        return np.zeros(6, dtype=np.float64)

    # Crop to exact block multiples
    cropped = gray[: h_blocks * block_size, : w_blocks * block_size]

    # Compute DCT on blocks
    ac_energies = []
    dc_values = []
    for i in range(0, cropped.shape[0], block_size):
        for j in range(0, cropped.shape[1], block_size):
            block = cropped[i : i + block_size, j : j + block_size].astype(np.float64)
            dct_block = cv2.dct(block)
            dc_values.append(dct_block[0, 0])
            ac_energy = np.sum(dct_block ** 2) - dct_block[0, 0] ** 2
            ac_energies.append(ac_energy)

    ac_energies = np.array(ac_energies)
    dc_values = np.array(dc_values)

    features = np.array([
        np.mean(ac_energies),
        np.std(ac_energies),
        np.mean(dc_values),
        np.std(dc_values),
        np.median(ac_energies),
        float(np.percentile(ac_energies, 90) - np.percentile(ac_energies, 10)),
    ], dtype=np.float64)

    return features


def _motion_features(gray1: np.ndarray, gray2: np.ndarray) -> np.ndarray:
    """Compute motion-related features between consecutive frames."""
    diff = np.abs(gray2.astype(np.float64) - gray1.astype(np.float64))
    return np.array([
        np.mean(diff),
        np.std(diff),
        np.max(diff),
    ], dtype=np.float64)


class VBLIINDSModule(PipelineModule):
    name = "vbliinds"
    description = "V-BLIINDS blind NR-VQA via DCT-domain NSS (Saad 2013)"
    default_config = {
        "subsample": 8,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._model = None
        self._backend = "heuristic"

    def setup(self) -> None:
        # Tier 1: Try native V-BLIINDS package
        try:
            import vbliinds
            self._model = vbliinds
            self._backend = "native"
            logger.info("V-BLIINDS (native) initialised")
            return
        except ImportError:
            pass

        # Tier 2: Heuristic fallback
        self._backend = "heuristic"
        logger.info("V-BLIINDS (heuristic) initialised — install vbliinds for full model")

    def process(self, sample: Sample) -> Sample:
        try:
            if self._backend == "native":
                score = float(self._model.predict(str(sample.path)))
            else:
                score = self._process_heuristic(sample)

            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.vbliinds_score = score

        except Exception as e:
            logger.warning(f"V-BLIINDS failed for {sample.path}: {e}")

        return sample

    def _process_heuristic(self, sample: Sample) -> Optional[float]:
        """Heuristic: DCT coefficient statistics + motion features → SVR-like scoring."""
        frames_gray = []

        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            try:
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total <= 0:
                    return None
                indices = np.linspace(0, total - 1, min(self.subsample, total), dtype=int)
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        frames_gray.append(
                            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
                        )
            finally:
                cap.release()
        else:
            img = cv2.imread(str(sample.path))
            if img is None:
                return None
            frames_gray.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64))

        if not frames_gray:
            return None

        # DCT features per frame
        dct_features_list = []
        for gray in frames_gray:
            dct_features_list.append(_dct_block_features(gray))

        dct_mean = np.mean(dct_features_list, axis=0)

        # AC energy: lower = less texture = potentially blurry
        ac_norm = min(dct_mean[0] / 5000.0, 1.0)
        # DC consistency: lower std = more uniform brightness
        dc_consistency = 1.0 / (1.0 + dct_mean[3] * 0.01)
        # AC range: broader distribution = more detail
        ac_range = min(dct_mean[5] / 10000.0, 1.0)

        spatial_quality = 0.45 * ac_norm + 0.25 * dc_consistency + 0.30 * ac_range

        # Motion features (temporal)
        if len(frames_gray) > 1:
            motion_feats = []
            for i in range(len(frames_gray) - 1):
                motion_feats.append(_motion_features(frames_gray[i], frames_gray[i + 1]))
            motion_arr = np.array(motion_feats)
            # Temporal stability: low variance in motion = smooth
            temporal = 1.0 / (1.0 + np.var(motion_arr[:, 0]) * 0.01)
        else:
            temporal = 1.0

        score = 0.7 * spatial_quality + 0.3 * temporal
        return float(np.clip(score, 0.0, 1.0))
